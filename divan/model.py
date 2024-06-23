import torch, random, warnings
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR

from psutil import cpu_count
from tqdm.auto import tqdm
import logging, yaml, time, inspect
import numpy as np

from divan.check import *
from divan.utils import *
from divan.module import *

warnings.simplefilter("ignore")
scales_ls = ['n', 's', 'm', 'l', 'x']
state_PATH = 'HGNetv2.pt'

logging.getLogger('asyncio').setLevel(logging.DEBUG)

block_name = 'Model_Manager'
FORMAT = '%(message)s'

setting_name = (
    "device",
    "amp_use",
    "scaler_use",
    "Mosaic_use",
    "EMA_use",
    "label_smooth",
)

train_name = (
    "Epoch",
    "GPU_mem",
    "Train_loss",
    "Size",
)

eval_name = (
    " ",
    " ",
    "Eval_loss",
    "Eval_acc",
)

class base_model(nn.Module):
    def __init__(self,
                 model_setting,
                 channels='RGB',
                 channels_mode='smooth',
                 seed=123):
        super().__init__()
        assert isinstance(channels, (str, type(None)))
        channels = channels.replace(' ', '') if channels != None else channels

        assert channels_mode in ['smooth', 'hard', 'auto']
        assert 1 <= len(channels) <= 3 or channels == 'auto'

        self.epoch = 1
        self.model_setting = model_setting
        self._set_seed(seed)
        if model_setting.endwith('.pt'):
            self._load_state(model_setting)

        self._read_model()

    def _ready_device(self):
        self.device_use = f'cuda' if torch.cuda.is_available() and self.device != 'cpu' else 'cpu'
        self.cuda_idx = f':{self.cuda_idx}' if self.cuda_idx is not None and self.device_use != 'cpu' else ''
        self.device = chose_cuda(self.device_use) if self.cuda_idx == '' else f'{self.device_use}{self.cuda_idx}'

    def _build_dataset(self):
        self.train_Manager = MiniImageNetDataset(f'dataset/{self.dataset_path}', 'train.txt',
                                                 transform=self.transform, channels=self.channels, channels_mode=self.channels_mode, silence=self.silence, shuffle=True, num_workers= self.num_workers, pin_memory=self.pin_memory, RAM=self.RAM, batch_size=self.batch_size, cutmix_p=self.cutmix_p)
        self.test_Manager = MiniImageNetDataset(f'dataset/{self.dataset_path}', 'check.txt',
                                                transform=self.transform, channels=self.channels, channels_mode=self.channels_mode, silence=True, shuffle=False, num_workers= self.num_workers, pin_memory=self.pin_memory, RAM=self.RAM, batch_size=self.batch_size)
        self.val_Manager = MiniImageNetDataset(f'dataset/{self.dataset_path}', 'val.txt',
                                               transform=self.transform, channels=self.channels, channels_mode=self.channels_mode, silence=True, shuffle=False, num_workers= self.num_workers, pin_memory=self.pin_memory, RAM=self.RAM, batch_size=self.batch_size)

        self._build_loader()

    def close_cutmix(self):
        self.train_Manager.close_cutmix()
        self.test_Manager.close_cutmix()
        self.val_Manager.close_cutmix()

    def _build_loader(self):
        self.train_loader = self.train_Manager.DataLoader
        self.test_loader = self.test_Manager.DataLoader
        self.val_loader = self.val_Manager.DataLoader

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
        #
        _desc = self.train_string if _training else self.eval_string
        correct, total = 0, 0
        tol_loss = 0

        pbar = tqdm(dataloader, desc=self.table_with_fix(_desc(epoch)), ncols=110)
        for _idx, (img, label) in enumerate(pbar):
            self.optimizer.zero_grad()
            img = img.float().to(self.device, non_blocking=dataloader.pin_memory)
            label = label.to(self.device, non_blocking=dataloader.pin_memory)
            epoch = torch.inf if isinstance(epoch, str) else epoch
            _epoch = 'eval' if epoch == torch.inf else epoch

            with torch.autocast(device_type=self.device_use, enabled=self.amp, dtype=torch.float16):
                with torch.set_grad_enabled(_training):
                    module = self.ema if not _training and self.ema and epoch >= self.ema_start else self.model
                    out = module(img)
                    loss = self.loss_function(out, label)

            tol_loss += loss.item()

            if _training:
                pbar.set_description(self.table_with_fix(_desc(_epoch, tol_loss / (_idx+1))))

                if self.amp and self.scaler != None:
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
                pbar.set_description(self.table_with_fix((_desc(_epoch, tol_loss / (_idx + 1), correct/total))))

        if _training:
            self.train_loss = tol_loss / (_idx + 1)
        else:
            self.eval_loss = tol_loss / (_idx + 1)

    def _train_step(self, epoch):
        torch.set_grad_enabled(True)

        logging.warning(f'\n{self.table_with_fix(train_name)}')
        self.model.train()
        self._loader(self.train_loader, epoch)

        logging.warning(f'{self.table_with_fix(eval_name)}')
        self.model.eval()
        self._loader(self.val_loader, epoch)

    def _test_step(self):
        logging.warning(f'{self.table_with_fix(train_name)}')
        self.model.train()
        self._loader(self.train_loader, 'val')

        logging.warning(f'{self.table_with_fix(eval_name)}')
        self.model.eval()
        self._loader(self.test_loader, 'val')

    def _save_state(self, epoch):
        self.state_dict = {'model_setting':self.model_setting,
                           'epoch':epoch,
                           'model':self.model.state_dict(),
                           'optimizer':self.optimizer.state_dict(),
                           'scheduler':self.scheduler.state_dict(),
                           'ema':self.ema.state_dict(),
                           'ema_scheduler':self.swa_scheduler.state_dict()
                           }

        torch.save(self.state_dict, self.state_PATH)

    def _load_state(self, state_PATH=None):
        if hasattr(self, self.state_dict) or state_PATH is not None:
            if state_PATH is not None:
                self.state_dict = torch.load(state_PATH)

            self.model_setting = self.state_dict['model_setting']
            self.epoch = self.state_dict['epoch']
            self.model = self.model.load_state_dict(self.state_dict['model'])
            self.optimizer = self.optimizer.load_state_dict(self.state_dict['optimizer'])
            self.scheduler = self.optimizer.load_state_dict(self.state_dict['scheduler'])
            self.ema = self.ema.load_state_dict(self.state_dict['ema'])
            self.ema_scheduler = self.ema_scheduler.load_state_dict(self.state_dict['ema_scheduler'])
        else:
            raise IOError


    def fit(self,
            dataset_path,
            epochs,
            size=224,
            silence=True,
            lr=0.005,
            batch_size=16,
            device=None,
            RAM=True,
            pin_memory=False,
            amp=True,
            ema=True,
            ema_start=0,
            early=20,
            cutmix_close=15,
            label_smoothing=0.1,
            transform=None,
            cuda_num=-1,
            cuda_idx=None,
            num_workers=0,
            cutmix_p=0,
            pretrained=False,
            ):
        assert isinstance(cuda_idx, (int, type(None)))

        log_filename = "all_log.log"
        logging.basicConfig(
            level=logging.WARNING if silence else logging.DEBUG,
            format=FORMAT,
            datefmt='%Y-%m-%d %H:%M',
            handlers=[
                logging.FileHandler(log_filename),  # 添加 FileHandler 寫入檔案
                logging.StreamHandler()  # 保持控制台輸出
            ]
        )
        logging.warning(f"{block_name}: Setting up")
        self._self_arg()
        self.in_channels = len(self.channels) if channels_mode == 'hard' else 3
        self.num_workers = num_workers if num_workers >= 0 else cpu_count(logical=False)
        self.state_dict = None
        self.state_PATH = state_PATH

        test_data(self.dataset_path)
        self._ready_device()
        self._build_dataset()
        self._test_ram_loader(self.train_loader)

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=0)

        self.ema_use = self.ema
        self.ema = AveragedModel(self.model).to(self.device)

        try:
            self.scaler = getattr(torch, self.device_use).amp.GradScaler(enabled=self.amp)
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

        logging.warning(f'{block_name}: Start training...\ntrain epochs - {epochs}\ninput channels - {self.input_c}')
        setup_value = (f"{self.device}",
                       f"{self.amp}",
                       f"{self.scaler != None}",
                       f"{self.cutmix_p > 0}",
                       f"{self.ema_use}",
                       f"{self.label_smoothing}"
                       )

        logging.warning(logging_table(setting_name,
                                      setup_value,
                                      table_name='Training setting',
                                      it='',
                                      min_ncol=11
                                      ))

        for epoch in range(self.epoch, epochs+1):
            self._train_step(epoch)

            if self.eval_loss < self.best_loss:
                self.best_loss = self.eval_loss
                self._save_state(epoch)
                self.early_count = 0
                self.best_epoch = epoch

            else:
                self.early_count += 1

            if epoch + self.cutmix_close == epochs + 1 and self.cutmix_p > 0:
                logging.info(f'{block_name}: Close cutmix')
                self.close_cutmix()

            if self.early_count >= self.early_lim:
                break

        logging.warning(f'{block_name}: Training ended - Best_epoch: {self.best_epoch}')
        self.valid()

    def valid(self):
        if self.state_dict != None:
            self._test_step()
        else:
            logging.warning(f'{block_name}: Model not training yet')

    @staticmethod
    def table_with_fix(col, fix_len=13):
        formatted_str = '|'.join(f'{x:^{fix_len}}' for x in col)
        return f'|{formatted_str}|'

    def train_string(self, epoch, loss=np.inf):
        _col_name = (
            f"{epoch}/{self.epochs}",
            f"{round(torch.cuda.memory_allocated(device = self.device)/1000000000, 2) if self.device_use == 'cuda' else 0.0}G",
            f"{round(loss, 3) if loss != np.inf else loss}",
            f"{self.size}",
        )

        return _col_name

    def eval_string(self, epoch, loss=np.inf, acc=0):
        _col_name = (
            f" ",
            f" ",
            f"{round(loss, 3) if loss != np.inf else loss}",
            f"{round(acc,3)}",
        )

        return _col_name

    def _self_arg(self):
        frame_func = inspect.currentframe().f_back
        args, _, _, values = inspect.getargvalues(frame_func)
        args.remove('self')

        for arg in args:
            setattr(self, arg, values[arg])

    def _test_ram_loader(self, loader):
        if self.RAM:
            start = time.monotonic()
            for data in tqdm(loader, desc=f"|{block_name} - speed check|"):
                _ = data
            logging.debug(f"{block_name}: RAM loader loop time: {round(time.monotonic()-start, 2)}s")

    @staticmethod
    def _set_seed(seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


class DIVAN_torch(base_model):
    def _read_model(self):
        assert self.train_Manager.class_num == self.test_Manager.class_num == self.val_Manager.class_num

        self.model = torch_model(self.model_setting, self.pretrained)
        if self.channels_mode != 'auto':
            self.model = inlayer_resize(self.model, in_channels=self.in_channels)
            self._fc_layer_resize()
            self.input_c = len(self.channels)

        else:
            m = Pool_Conv()
            self.model = inlayer_resize(self.model, in_channels=1)
            self.model = nn.Sequential(*[m, self.model]).to(self.device)
            self._fc_layer_resize()
            self.input_c = 'auto'


class DIVAN(base_model):
    def _read_model(self):
        assert self.train_Manager.class_num == self.test_Manager.class_num == self.val_Manager.class_num
        self.yaml = self._read_yaml(self.model_setting.split('.yaml')[0])
        self.yaml['nc'] = self.train_Manager.class_num
        intput_channels = len(self.channels) if self.channels_mode != 'auto' else self.channels_mode

        self.model = yaml_model(self.yaml, intput_channels, self.device)

        if self.channels_mode != 'auto':
            self.model = inlayer_resize(self.model, self.in_channels)
            if self.model.fc_resize:
                self._fc_layer_resize()
            else:
                self.model = self.model.to(self.device)
            self.input_c = len(self.channels)

        else:
            if self.model.fc_resize:
                self._fc_layer_resize()
            else:
                self.model = self.model.to(self.device)
            self.input_c = 'auto'

    @staticmethod
    def _read_yaml(yaml_path):
        logging.warning(f'{block_name}: yaml Path - {yaml_path}')
        with open(f'cfg/{yaml_path}.yaml', 'r', encoding="utf-8") as stream:
            return yaml.safe_load(stream)

#if __name__ == "__main__":
#    FORMAT = '[%(levelname)s] | %(asctime)s | %(message)s'
#    FORMAT = '%(message)s'
#    logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt='%Y-%m-%d %H:%M')

    #model = base_CV().train()
#    model.train(70)


##org.update(org.fromkeys(['x','x1'], 1))