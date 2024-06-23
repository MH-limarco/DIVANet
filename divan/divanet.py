import torch, os, random, warnings

from tqdm.auto import tqdm
import logging, yaml, time
import numpy as np
import pandas as pd

from divan.check import *
from divan.utils import *
from divan.module import *
from divan.parse_task import *
from divan.scheduler import *

warnings.simplefilter("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True

class model_Manager(nn.Module):
    def __init__(self, model_setting,
                 channels='RGB',
                 random_p=0.8,
                 fix_mean=False,
                 cut_channels=False,
                 amp=True,
                 device='cuda',
                 seed=0
                 ):
        super().__init__()
        apply_config(self, __file__)
        apply_args(self)
        self.model_weight = False
        self.train_PATH = self.save_PATH + f"train"
        self.test_PATH = self.save_PATH + f"test"

        if not os.path.exists(self.train_PATH):
            os.makedirs(self.train_PATH)

        self.set_seed(seed)
        if model_setting.endswith('.pt'):
            self._load_state(model_setting)
            #train/{self.model_setting.split('.')[0]}/weight/best.pt
        else:
            self._read_model()
    def to(self, device):
        assert hasattr(self, 'model')
        self.device, self.device_use = self._ready_device(device)

        self.model = self.model.to(self.device)
        if hasattr(self, "optimizer"):
            self.optimizer_to(self.optimizer, self.device)

    def forward(self, x):
        pass
        with torch.autocast(device_type=self.device):
            return self.model(x.float().to(self.device))

    def fit(self, dataset_path, epochs,
            label_path=["train.txt", "val.txt", "test.txt"],
            size=224,
            batch_size=128,
            use_compile=False,
            #optim
            lr=1e-3,
            weight_decay=0.01,
            # scheduler
            warnup_step=0,
            warnup_start_factor=0.1,
            T_0=3,
            T_mult=2,
            eta_min=0,
            endstep_epochs=0,
            endstep_start_factor=1,
            endstep_patience=5,
            endstep_factor=0.1,
            #train
            early_stopping=15,
            label_smoothing=0.1,
            silence=False,
            #dataset
            cutmix_p=1,
            pin_memory=False,
            shuffle=True,
            RAM=True,
            ncols=90,
            RAM_lim=0.925,
            num_workers=-1,
            ):
        apply_args(self)
        self._build_dataset()
        self.train_PATH = self.check_dirs(self.train_PATH, True)
        dataset_class_num = self.Dataset.class_num
        self.datafram_ls = []
        if hasattr(self, 'class_num'):
            if dataset_class_num != self.class_num:
                self.num_class = dataset_class_num
                self._fclayer_resize(self.num_class)
        else:
            self.num_class = dataset_class_num
            self._fclayer_resize(self.num_class)

        self.to(self.device)
        self.epoch, self.epochs = self.epoch if hasattr(self, "epoch") else [0, epochs]

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=self.lr,
                                               weight_decay=self.weight_decay,
                                               )

            self.scheduler = self.warn_up_scheduler()
            self.scheduler_end = self.end_step_scheduler()

        try:
            self.scaler = getattr(torch, self.device_use).amp.GradScaler(enabled=self.amp)
        except:
            self.scaler = None

        self.step_count = 0

        best_loss = float('inf')
        for epoch in range(self.epoch, self.epochs - endstep_epochs):
            self._training_step(epoch)
            self._save_state(epoch)
            if best_loss > self.eval_loss:
                best_loss = self.eval_loss
                self.best_epoch = epoch
                self._save_state(epoch, best=True)
                self.step_count = 0
            else:
                self.step_count += 1

            if self.step_count > self.early_stopping:
                break

        self.step_count = 0
        self.early_stopping = round(self.early_stopping / 2)

        if endstep_epochs > 0:
            if 'epoch' in locals():
                epoch += 1
                self._load_state()
                self.scheduler_end.init_lr()
            else:
                epoch = 0
                self.epochs = endstep_epochs

            if self.cutmix_p > 0:
                self.Dataset.close_cutmix()

            logging.info(f"{self.block_name}: Starting end_step...")
            for _epoch in range(epoch, epoch+endstep_epochs):
                self._training_step(_epoch, end_step=True)
                self._save_state(_epoch)
                if best_loss > self.eval_loss:
                    best_loss = self.eval_loss
                    self.best_epoch = _epoch
                    self._save_state(_epoch, best=True)
                    self.step_count = 0
                else:
                    self.step_count += 1

                if self.step_count > self.early_stopping:
                    break

        logging.info(f"{self.block_name}: Train finish - best epoch {self.best_epoch + 1}")

        self.step_count = 0
        self._load_state()
        self._testing_step()
        self.draw_plot()

    def valid(self):
        assert hasattr(self, "state_dict")

    def _read_model(self):
        raise NotImplementedError

    def _fclayer_resize(self, num_class):
        self.model = fclayer_resize(self.model, num_class)

    def _save_state(self, epoch, best=False):
        state_dict = {'model_setting': self.model_setting,
                      'num_class': self.num_class,
                      'epoch': [epoch, self.epochs],
                      'model_dict': self.model.state_dict(),
                      'optimizer_dict': self.optimizer.state_dict(),
                      'scheduler_dict': self.scheduler.state_dict(),
                      'scheduler_end_dict': self.scheduler_end.state_dict(),
                      }
        if best:
            getattr(self, "state_dict", state_dict)

        torch.save(state_dict, os.path.join(self.train_PATH, 'best.pt' if best else 'last.pt'))

    def _load_state(self, state_PATH=None):
        assert state_PATH.endswith('.pt') if isinstance(state_PATH, str) else state_PATH is None
        state_PATH = os.path.join(self.train_PATH, 'best.pt') if state_PATH is None else state_PATH
        self.state_dict = torch.load(state_PATH)
        apply_kwargs(self, self.state_dict)

        if not hasattr(self, 'model'):
            self._read_model()
            self.optimizer = torch.optim.AdamW(self.model.parameters())
        self._load_model()

    def _load_model(self):
        self.model.load_state_dict(self.model_dict)
        self.optimizer.load_state_dict(self.optimizer_dict)

        if not hasattr(self, "scheduler"):
            self.scheduler = self.warn_up_scheduler(True)
        if not hasattr(self, "scheduler_end"):
            self.scheduler_end = self.end_step_scheduler(True)

        self.scheduler.load_state_dict(self.scheduler_dict)
        self.scheduler_end.load_state_dict(self.scheduler_end_dict)
        self.model_weight = True

    def _ready_device(self, device=None):
        device = self.device.split(':') if device is None else device.split(':')
        device, cuda_idx = device if len(device) > 1 else [device[0], None]
        device = device if torch.cuda.is_available() and device != 'cpu' else 'cpu'
        cuda_idx = f':{cuda_idx}' if cuda_idx is not None and device != 'cpu' else cuda_idx
        torch.backends.cudnn.benchmark = True if device == 'cuda' else False
        return chose_cuda(device) if cuda_idx is None else f'{device}{cuda_idx}', device

    def _build_dataset(self):
        self.Dataset = Dataset_Manager(dataset_path=self.dataset_path,
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

    def _close_cutmix(self):
        self.Dataset.close_cutmix()

    def _run_loader(self, dataloader, epoch, end_step=False):
        epoch = torch.inf if isinstance(epoch, str) else epoch

        _epoch = 'eval' if epoch == torch.inf else epoch
        _training = self.model.training and _epoch != 'eval'
        _desc = self._train_string if _training else self._eval_string
        correct, total = 0, 0
        tol_loss = 0

        module = self.model
        pbar = tqdm(dataloader, desc=self.table_with_fix(_desc(epoch)), ncols=110)
        for _idx, (img, label) in enumerate(pbar):

            img = img.float().to(self.device, non_blocking=dataloader.pin_memory)
            label = label.to(self.device, non_blocking=dataloader.pin_memory)
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device_use, enabled=self.amp, dtype=torch.float16):
                with torch.set_grad_enabled(_training):
                    out = module(img)
                    loss = self.loss_function(out, label)
            tol_loss += loss.mean().item()
            correct += (torch.argmax(label, 1) == torch.argmax(out, 1)).sum().item()
            total += label.size(0)
            if _training:
                pbar.set_description(self.table_with_fix(_desc(_epoch, tol_loss / (_idx + 1))))
                if epoch != 'eval':
                    if self.amp and self.scaler != None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

            else:
                pbar.set_description(self.table_with_fix((_desc(end_step, tol_loss / (_idx + 1), correct / total))))

        if _training:
            if _epoch != 'eval':
                self.scheduler_end.step(tol_loss) if end_step else self.scheduler.step()

            self.train_loss = tol_loss / (_idx + 1)
        else:
            self.eval_loss = tol_loss / (_idx + 1)

        return tol_loss / (_idx + 1), correct / total,

    def _training_step(self, epoch, end_step=False):
        torch.set_grad_enabled(True)
        logging.warning(f'\n{self.table_with_fix(self.fit_training_col)}')
        self.model.train()
        train_loss, train_acc = self._run_loader(self.Dataset.train_loader, epoch, end_step)

        logging.warning(f'{self.table_with_fix(self.fit_eval_col)}')
        self.model.eval()
        eval_loss, eval_acc = self._run_loader(self.Dataset.val_loader, epoch, end_step)

        lr = round(self.optimizer.param_groups[0]['lr'], 6)

        self.datafram_ls.append([epoch, train_loss, train_acc, eval_loss, eval_acc, lr, end_step])

    def _testing_step(self):
        self.to(self.device)
        torch.set_grad_enabled(False)
        logging.warning(f'\n{self.table_with_fix(self.test_training_col)}')
        self.model.train()
        train_loss, train_acc = self._run_loader(self.Dataset.train_loader, 'val')

        logging.warning(f'{self.table_with_fix(self.fit_eval_col)}')
        self.model.eval()
        eval_loss, eval_acc = self._run_loader(self.Dataset.test_loader, 'val')

        lr = np.nan
        self.datafram_ls.append(['val', train_loss, train_acc, eval_loss, eval_acc, lr, True])

    def _train_string(self, epoch, loss=np.inf):
        return (f"{epoch+1}/{self.epochs}",
                f"{round(torch.cuda.memory_allocated(device = self.device)/1000000000, 2) if self.device_use == 'cuda' else 0.0}G",
                f"{round(loss, 3) if loss != np.inf else loss}",
                f"{self.size}"
                )

    def _eval_string(self, end_step, loss=np.inf, acc=0):
        return (f" ",
                f"{end_step}",
                f"{round(loss, 3) if loss != np.inf else loss}",
                f"{round(acc, 3)}")

    def warn_up_scheduler(self, free=False):
        scheduler_warn_up = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                              start_factor=self.warnup_start_factor if not free else 1,
                                                              total_iters=self.warnup_step if not free else 1)

        scheduler_start = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                               T_0=self.T_0 if not free else 1,
                                                                               T_mult=self.T_mult if not free else 1,
                                                                               eta_min=self.eta_min if not free else 1,
                                                                               last_epoch=-1)
        return torch.optim.lr_scheduler.SequentialLR(self.optimizer,
                                                     [scheduler_warn_up, scheduler_start],
                                                     milestones=[self.warnup_step if not free else 1],
                                                     )

    def end_step_scheduler(self, free=False):
        return FactorReduceLROnPlateau(self.optimizer,
                                       start_factor=self.endstep_start_factor if not free else 0,
                                       patience=self.endstep_patience if not free else 0,
                                       factor=self.endstep_factor if not free else 0,
                                       )

    def draw_plot(self):
        df = pd.DataFrame(self.datafram_ls, columns=['epoch', 'train_loss', 'train_acc', 'eval_loss', 'eval_acc', 'lr', 'end_step']).set_index('epoch')
        PATH = self.train_PATH.replace('weight','')
        train_df, test_df = df.iloc[:-1], df.iloc[-1]

        ax_loss = train_df[['train_loss', 'eval_loss']].plot()
        ax_acc = train_df[['train_acc', 'eval_acc']].plot()
        ax_lr = train_df[["lr"]].plot()

        logging.info(f"{self.block_name}: result_PATH - {PATH}")
        ax_loss.get_figure().savefig(PATH + "loss.png")
        ax_acc.get_figure().savefig(PATH + "acc.png")
        ax_lr.get_figure().savefig(PATH + "lr.png")
        df.to_csv(PATH+"result.csv")

    def check_dirs(self, PATH, weight=False):
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        i = 1
        while os.path.exists(os.path.join(PATH, f"{self.model_setting.split('.')[0]}-{i}")):
            i += 1

        new_PATH = os.path.join(PATH, f"{self.model_setting.split('.')[0]}-{i}")
        new_PATH = os.path.join(new_PATH, "weight") if weight else new_PATH

        os.makedirs(new_PATH)
        return new_PATH

    def optimizer_to(self, optimizer=None, device=None):
        optimizer = self.optimizer if optimizer is None else optimizer
        device = self.device if device is None else device

        for param in optimizer.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to()
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

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
        self.yaml = self._read_yaml(self.model_setting.split('.yaml')[0])
        self.yaml["nc"] = self.num_class if hasattr(self, "num_class") else self.yaml["nc"]
        intput_channels = len(self.channels) if self.channels != 'auto' else self.channels

        self.model = Divanet_model(self.yaml, intput_channels)
        try:
            if self.use_compile:
                start_time = time.monotonic()
                self.model = torch.compile(self.model)
                logging.info(f"{self.block_name}: Compile time - {round(time.monotonic()-start_time, 5)}s")
        except:
            pass

    def _read_yaml(self, yaml_path):
        with open(f'{self.cfg_PATH}{yaml_path}.yaml', 'r', encoding="utf-8") as stream:
            return yaml.safe_load(stream)


if __name__ == "__main__":
    pass
    #model_Manager()