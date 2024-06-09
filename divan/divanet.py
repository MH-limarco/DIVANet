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

block_name = 'Model_Manager'
FORMAT = '%(message)s'

warnings.simplefilter("ignore")
logging.getLogger('asyncio').setLevel(logging.DEBUG)
scales_ls = ['n', 's', 'm', 'l', 'x']
state_PATH = 'HGNetv2.pt'

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

class base_Manager(nn.Module):
    def __init__(self):
        pass

    def to(self):
        pass

    def forward(self, x):
        pass
        with torch.autocast(device_type=self.device):
            return self.model(x.float().to(self.device))

    def fit(self):
        pass

    def valid(self):
        pass

    def _fc_layer_resize(self):
        pass

    def _read_model(self):
        raise NotImplementedError

    def _save_state(self, epoch):
        pass

    def _load_state(self, state_PATH=None):
        pass

    def _train_ready_device(self):
        pass

    def _train_build_dataset(self):
        pass

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
                f"{round(acc,3)}")

    def _self_arg(self):
        frame_func = inspect.currentframe().f_back
        args, _, _, values = inspect.getargvalues(frame_func)
        args.remove('self')

        for arg in args:
            setattr(self, arg, values[arg])

    def loader_perf(self, loader):
        start = time.monotonic()
        for data in tqdm(loader, desc=f"|{block_name} - speed test|"):
            _ = data
        logging.debug(f"{block_name}: RAM loader loop time: {round(time.monotonic()-start, 2)}s")

    @staticmethod
    def table_with_fix(col, fix_len=13):
        formatted_str = '|'.join(f'{x:^{fix_len}}' for x in col)
        return f'|{formatted_str}|'

    @staticmethod
    def set_seed(seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
