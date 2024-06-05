import torch, os, random, warnings
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.optim.swa_utils import AveragedModel, SWALR

from psutil import cpu_count
from tqdm.auto import tqdm
import logging, yaml
import numpy as np

from src.test import *
from src.utils import *
from src.module import *


warnings.simplefilter("ignore")
scales_ls = ['n', 's', 'm', 'l', 'x']
memory_used_command = "nvidia-smi --query-gpu=memory.used --format=csv"

block_name = 'Model_Manager'
FORMAT = '%(message)s'

class base_CV(nn.Module):
    def __init__(self,

                 dataset_path,
                 silence=True,

                 size=224,
                 channels='RGB',
                 channels_cut='smooth',
                 lr=0.005,
                 batch_size=16,
                 device=None,
                 RAM=True,
                 pin_memory=False,
                 amp=True,
                 ema=True,
                 ema_start=0,
                 early=20,
                 label_smoothing=0.075,

                 transform=None,
                 cuda_num=-1,
                 cuda_idx=1,
                 num_workers=-1,
                 cutmix_p=0,
                 seed=123
                 ):
        super().__init__()
        assert type(channels) in [list, str]
        assert channels_cut in ['smooth', 'hard']
        channels = channels.replace(' ','')
        assert 1 <= len(channels) <=3

        self.dataset_path = dataset_path
        self.silence = silence
        self.channels = channels
        self.channels_cut = channels_cut
        self.in_channels = len(self.channels) if channels_cut == 'hard' else 3

        logging.basicConfig(level=logging.WARNING if silence else logging.DEBUG,format=FORMAT,datefmt='%Y-%m-%d %H:%M')
        test_data(self.dataset_path)

        self.lr = lr
        self.label_smoothing = label_smoothing
        self.size = size
        self.ema_use = ema
        self.amp_use = amp
        self.ema_start = ema_start
        self.early_lim = early

        self.transform = transform
        self.batch_size = batch_size
        self.cutmix_p = cutmix_p
        self.RAM = RAM
        self.pin_memory = pin_memory

        self.device = device
        self.cuda_idx = cuda_idx
        self.cuda_num = cuda_num

        self.num_workers = num_workers if num_workers > 0 else cpu_count(logical = True)
        self._set_seed(seed)

        self.state_dict = None
        self.state_PATH = 'save.pt'

        self._ready_device()
        self._build_dataset()

    def _ready_device(self):
        self.device_use = f'cuda' if torch.cuda.is_available() and self.device != 'cpu' else 'cpu'
        self.device = f'cuda:{self.cuda_idx}' if torch.cuda.is_available() and self.device != 'cpu' else 'cpu'

        if self.device_use == f'cuda:' and not 0 <= self.cuda_num < 2:
            device_num = torch.cuda.device_count() if self.cuda_num < 0 else self.cuda_num
            self.device_ls = [f'cuda:{i}' for i in range(device_num)]
        else:
            self.device_ls = []

    def _build_dataset(self):
        self.train_Manager = MiniImageNetDataset(f'dataset/{self.dataset_path}', 'train.txt',
                                                 transform=self.transform, channels=self.channels, channels_cut=self.channels_cut, silence=self.silence, shuffle=True, pin_memory=self.pin_memory, RAM=self.RAM, batch_size=self.batch_size, cutmix_p=self.cutmix_p)
        self.test_Manager = MiniImageNetDataset(f'dataset/{self.dataset_path}', 'test.txt',
                                                transform=self.transform, channels=self.channels, channels_cut=self.channels_cut, silence=True, shuffle=False, pin_memory=self.pin_memory, RAM=self.RAM, batch_size=self.batch_size)
        self.val_Manager = MiniImageNetDataset(f'dataset/{self.dataset_path}', 'val.txt',
                                               transform=self.transform, channels=self.channels, channels_cut=self.channels_cut, silence=True, shuffle=False, pin_memory=self.pin_memory, RAM=self.RAM, batch_size=self.batch_size)

        self._build_loader()

    def _close_mosaic(self):
        self.train_Manager.close_mosaic()
        self.test_Manager.close_mosaic()
        self.val_Manager.close_mosaic()

    def _build_loader(self):
        self.train_loader = self.train_Manager.DataLoader
        self.test_loader = self.test_Manager.DataLoader
        self.val_loader = self.val_Manager.DataLoader

    def _model_resize(self, in_channels=3):
        for name, layer in list(self.model.named_modules()):
            if isinstance(layer, nn.Conv2d):
                out_channels = layer.out_channels
                kernel_size = layer.kernel_size
                new_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

                name_parts = name.split('.')
                sub_module = self.model
                for part in name_parts[:-1]:
                    sub_module = getattr(sub_module, part)
                setattr(sub_module, name_parts[-1], new_layer)
                break
        self._fc_layer_resize()

    def _fc_layer_resize(self):
        for name, layer in reversed(list(self.model.named_modules())):
            if isinstance(layer, nn.Linear):
                input_features = layer.in_features
                new_layer = nn.Linear(input_features, self.train_Manager.class_num)

                name_parts = name.split('.')
                sub_module = self.model
                for part in name_parts[:-1]:
                    sub_module = getattr(sub_module, part)
                setattr(sub_module, name_parts[-1], new_layer)
                break

        self.model = self.model.to(self.device)

    def _read_model(self):
        raise NotImplementedError

    def forward(self, x):
        with torch.autocast(device_type=self.device):
            return self.model(x.float().to(self.device))

    def _loader(self, dataloader, epoch):
        _training = self.model.training
        _str_desc = self.train_string if _training else self.eval_string
        correct, total = 0, 0
        tol_loss = 0

        pbar = tqdm(dataloader, desc=_str_desc(epoch), ncols=110)
        for _idx, (img, label) in enumerate(pbar):
            self.optimizer.zero_grad()
            img = img.float().to(self.device, non_blocking=dataloader.pin_memory)
            label = label.to(self.device, non_blocking=dataloader.pin_memory)
            epoch = torch.inf if isinstance(epoch, str) else epoch
            _epoch = 'eval' if epoch == torch.inf else epoch

            with torch.autocast(device_type=self.device_use, enabled=self.amp_use, dtype=torch.float16):
                with torch.set_grad_enabled(_training):
                    module = self.ema if not _training and self.ema_use and epoch >= self.ema_start else self.model
                    out = module(img)
                    loss = self.loss_function(out, label)

            tol_loss += loss.item()

            if _training:
                pbar.set_description(_str_desc(_epoch, tol_loss / (_idx+1)))

                if self.amp_use and self.scaler != None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if epoch >= self.ema_start:
                    self.ema.update_parameters(module)
                    self.swa_scheduler.step()
                else:
                    self.scheduler.step()
            else:
                correct += (torch.argmax(label,1) == torch.argmax(out, 1)).sum().item()
                total += label.size(0)
                pbar.set_description(_str_desc(_epoch, tol_loss / (_idx + 1), correct/total))

        if _training:
            self.train_loss = tol_loss / (_idx + 1)
        else:
            self.eval_loss = tol_loss / (_idx + 1)

    def _train_step(self, epoch):
        torch.set_grad_enabled(True)

        logging.warning(f'{self.train_col_string()}')
        self.model.train()
        self._loader(self.train_loader, epoch)

        logging.warning(f'{self.eval_col_string()}')
        self.model.eval()
        self._loader(self.val_loader, epoch)



    def _test_step(self):
        logging.warning(f'{self.train_col_string()}')
        self.model.train()
        self._loader(self.train_loader, 'val')

        logging.warning(f'{self.eval_col_string()}')
        self.model.eval()
        self._loader(self.test_loader, 'val')

    def _save_state(self):
        self.state_dict = {'model':self.model.state_dict(),
                           'optimizer':self.optimizer.state_dict(),
                           'scheduler':self.scheduler.state_dict(),
                           'ema':self.ema.state_dict(),
                           'ema_scheduler':self.swa_scheduler.state_dict()
                           }

        torch.save(self.state_dict, self.state_PATH)

    def train(self, epochs):
        self._read_model()

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=0)

        self.ema = AveragedModel(self.model).to(self.device)

        try:
            self.scaler = getattr(torch, self.device_use).amp.GradScaler(enabled=self.amp_use)
        except:
            self.scaler = None

        self.best_loss = np.inf
        self.best_epoch = 0

        self.epochs = epochs
        self.swa_start = 0
        self.early_lim = 20

        self.early_count = 0

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.lr/25, last_epoch=-1)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.lr*1.5, anneal_strategy="cos")

        logging.warning(f'{block_name}: Start training...')
        logging.warning(self.setup_col_string())
        logging.warning(self.setup_string())

        for epoch in range(1, epochs+1):
            self._train_step(epoch)

            if self.eval_loss < self.best_loss:
                self.best_loss = self.eval_loss
                self._save_state()
                self.early_count = 0
                self.best_epoch = epoch

            else:
                self.early_count += 1

            if self.early_count >= self.early_lim:
                break

        logging.WARNING(f'{block_name}: Training ended - Best_epoch: {self.best_epoch}')
        self.valid()

    def valid(self):
        if self.state_dict != None:
            self._test_step()
        else:
            logging.WARNING(f'{block_name}: Model not training yet')

    def predict(self, x):
        return self._forward(x)

    def setup_col_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        _col_name = (
            "device",
            "amp_use",
            "scaler_use",
            "Mosaic_use",
            "EMA_use",
            "label_smooth_rate",
        )
        return ('Training setting:\n'+f"{'|{:^21s}'* (len(_col_name))}|").format(*_col_name)

    def train_col_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        _col_name = (
            "Epoch",
            "GPU_mem",
            "Train_loss",
            "Size",
        )
        return ('\n'+f"{'|{:^13s}'* (len(_col_name))}|").format(*_col_name)

    def eval_col_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        _col_name = (
            " ",
            " ",
            "Eval_loss",
            "Eval_acc",
        )
        return (f"{'|{:^13s}'* (len(_col_name))}|").format(*_col_name)

    def setup_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        _col_name = (
            f"{self.device}",
            f"{self.amp_use}",
            f"{self.scaler != None}",
            f"{self.cutmix_p > 0}",
            f"{self.ema_use}",
            f"{self.label_smoothing}",
            #f"{self.loss_function.__class__.__name__}",
        )
        return (f"{'|{:^21s}'* (len(_col_name))}|").format(*_col_name)

    def train_string(self, epoch, loss=np.inf):
        _col_name = (
            f"{epoch}/{self.epochs}",
            f"{round(torch.cuda.memory_allocated(device = self.device)/1000000000, 2) if self.device_use == 'cuda' else 0.0}G",
            f"{round(loss, 3) if loss != np.inf else loss}",
            f"{self.size}",
        )

        return (f"{'|{:^13s}'* (len(_col_name))}|").format(*_col_name)

    def eval_string(self, epoch, loss=np.inf, acc=0):
        _col_name = (
            f" ",
            f" ",
            f"{round(loss, 3) if loss != np.inf else loss}",
            f"{round(acc,3)}",
        )

        return (f"{'|{:^13s}'* (len(_col_name))}|").format(*_col_name)
    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class Torch_CV(base_CV):
    def __init__(self, model_name, *args, pretrained=None, **kwargs):
        super(Torch_CV, self).__init__(*args, **kwargs)
        self.model_name = model_name
        self.pretrained = pretrained


    def _read_model(self):
        assert self.train_Manager.class_num == self.test_Manager.class_num == self.val_Manager.class_num

        self.model = torch_model(self.model_name, self.pretrained)
        self._model_resize(in_channels=self.in_channels)

class Custom_CV(base_CV):
    def __init__(self, yaml, *args, **kwargs):
        super(Custom_CV, self).__init__(*args, **kwargs)
        assert yaml.endswith('.yaml')
        assert len(yaml.split('.yaml')) == 2

        self.yaml = yaml.split('.yaml')[0]
        print(self.yaml)

    def _read_model(self):
        assert self.train_Manager.class_num == self.test_Manager.class_num == self.val_Manager.class_num
        self.yaml = self._read_yaml(self.yaml)
        self.yaml['nc'] = self.train_Manager.class_num

        self.model = yaml_model(self.yaml, len(self.channels), self.device)
        self._model_resize(in_channels=self.in_channels)


    @staticmethod
    def _read_yaml(yaml_path):
        logging.warning(f'{block_name}: yaml Path - {yaml_path}')
        with open(f'cfg/{yaml_path}.yaml', 'r', encoding="utf-8") as stream:
            return yaml.safe_load(stream)





#if __name__ == "__main__":
#    FORMAT = '[%(levelname)s] | %(asctime)s | %(message)s'
#    FORMAT = '%(message)s'
#    logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt='%Y-%m-%d %H:%M')

#    model = Custom_CV('test123.yaml', 'dataset', cuda_num=-1, cutmix_p=0)
#    model.train(70)


##org.update(org.fromkeys(['x','x1'], 1))