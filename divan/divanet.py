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
import logging, yaml, time, inspect
import numpy as np

from divan.check import *
from divan.utils import *
from divan.module import *
from divan.parse_task import *

warnings.simplefilter("ignore")
state_PATH = 'HGNetv2.pt'

#dataset = Dataset_Manager('dataset', batch_size=64, channels=[None, 'RG'], cut_channels=[True, False], fix_mean=[True, True], RAM=True, shuffle=True)

class model_Manager(nn.Module):
    def __init__(self, model_setting,
                 channels='RGB',
                 amp=True,
                 device='cuda',
                 seed=123
                 ):
        super().__init__()
        apply_config(self, __file__)
        apply_args(self)
        self.glob_amp = amp

        self.set_seed(seed)
        if model_setting.endswith('.pt'):
            self._load_state(model_setting)
            #train/{self.model_setting.split('.')[0]}/weight/best.pt
        else:
            self._read_model()
    def to(self, device):
        assert hasattr(self, 'model')
        device = self._ready_device(device)

        self.model = self.model.to(device)

        if hasattr(self, 'ema'):
            self.ema = self.ema.to(device)

    def forward(self, x):
        pass
        with torch.autocast(device_type=self.device):
            return self.model(x.float().to(self.device))

    def fit(self, ):
        pass

    def valid(self):
        pass

    def _read_model(self):
        raise NotImplementedError

    def _save_state(self, epoch):
        self.state_dict = {'model_setting': self.model_setting,
                           'num_class': self.num_class,
                           'epoch': epoch,
                           'model_dict': self.model.state_dict(),
                           'ema_dict': self.ema.state_dict(),
                           'optimizer_dict': self.optimizer.state_dict(),
                           'scheduler_dict': self.scheduler.state_dict(),
                           'ema_scheduler_dict': self.swa_scheduler.state_dict()
                           }

        torch.save(self.state_dict, self.state_PATH)

    def _load_state(self, state_PATH=None):
        assert state_PATH.endswith('.pt')
        state_PATH = self.state_PATH if state_PATH is None else state_PATH
        self.state_dict = torch.load(state_PATH)
        apply_kwargs(self, self.state_dict)

        self._read_model()
        self._load_model()
    def _load_model(self):
        self.model.load_state_dict(self.model_dict)
        self.ema.load_state_dict(self.ema_dict)
        self.optimizer.load_state_dict(self.optimizer_dict)
        self.scheduler.load_state_dict(self.scheduler_dict)
        self.swa_scheduler.load_state_dict(self.ema_scheduler_dict)

    def _ready_device(self, device=None):
        device = self.device.split(':') if device is None else device.split(':')
        device, cuda_idx = device if len(device) > 1 else [device, None]
        device = device if torch.cuda.is_available() and device != 'cpu' else 'cpu'
        cuda_idx = f':{cuda_idx}' if cuda_idx is not None and device != 'cpu' else cuda_idx
        return chose_cuda(device) if cuda_idx is None else f'{self._device}{self.cuda_idx}'


    def _build_dataset(self):
        Dataset = Dataset_Manager(dataset_path=self.dataset_path,
                                  label_path=self.label_path,
                                  channels=self.channels,
                                  size=self.size,
                                  batch_size=self.batch_size,
                                  pin_memory=self.pin_memory,
                                  shuffle=self.shuffle,
                                  silence=self.silence,
                                  fix_mean=self.fix_mean,
                                  cut_channels=self.cut_channels,
                                  random_p=self.random_p,
                                  num_workers=self.num_workers,
                                  RAM=self.RAM,
                                  cutmix_p=self.cutmix_p,
                                  ncols=self.ncols,
                                  RAM_lim=self.RAM_lim
                                  )

    def _train_close_cutmix(self):
        pass

    def _train_build_loader(self):
        pass

    def _train_run_loader(self):
        pass

    def _train_training_stop(self):
        pass

    def _train_testing_step(self):
        pass

    def _train_training_string(self, epoch, loss=np.inf):
        return (f"{epoch}/{self.epochs}",
                f"{round(torch.cuda.memory_allocated(device = self.device)/1000000000, 2) if self.device_use == 'cuda' else 0.0}G",
                f"{round(loss, 3) if loss != np.inf else loss}",
                f"{self.size}"
                )

    def _train_eval_string(self, loss=np.inf, acc=0):
        return (f" ",
                f" ",
                f"{round(loss, 3) if loss != np.inf else loss}",
                f"{round(acc, 3)}")

    def _self_arg(self):
        frame_func = inspect.currentframe().f_back
        args, _, _, values = inspect.getargvalues(frame_func)
        args.remove('self')

        for arg in args:
            setattr(self, arg, values[arg])

    def loader_perf(self, loader):
        start = time.monotonic()
        for data in tqdm(loader, desc=f"|{self.block_name} - speed test|"):
            _ = data
        logging.debug(f"{self.block_name}: RAM loader loop time: {round(time.monotonic()-start, 2)}s")

    @staticmethod
    def table_with_fix(col, fix_len=13):
        formatted_str = '|'.join(f'{x:^{fix_len}}' for x in col)
        return f'|{formatted_str}|'

    @staticmethod
    def set_seed(seed):
        if isinstance(seed, int):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

class DIVAN(model_Manager):
    def _read_model(self):
        self.label_smoothing = 0.1
        self.lr = 0.0005
        self.weight_decay = 0
        self.epochs = 0
        self.state_PATH = self.save_PATH + f"train/{self.model_setting.split('.')[0]}/weight/best.pt"

        self.yaml = self._read_yaml(self.model_setting.split('.yaml')[0])
        self.yaml["nc"] = self.num_class if hasattr(self, "num_class") else self.yaml["nc"]
        print(self.yaml["nc"])
        intput_channels = len(self.channels) if self.channels != 'auto' else self.channels

        self.model = Divanet_model(self.yaml, intput_channels)

        self.ema = AveragedModel(self.model)

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.lr/25, last_epoch=-1)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.lr*1.5, anneal_strategy="cos")

        try:
            self.scaler = getattr(torch, self.device_use).amp.GradScaler(enabled=self.amp)
        except:
            self.scaler = None

        self._save_state(0)


    def _read_yaml(self, yaml_path):
        with open(f'{self.cfg_PATH}{yaml_path}.yaml', 'r', encoding="utf-8") as stream:
            return yaml.safe_load(stream)


if __name__ == "__main__":
    pass
    #model_Manager()