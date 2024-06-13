from torch.utils.data import Dataset, DataLoader, default_collate
from torch.multiprocessing import set_sharing_strategy
from torchvision.transforms import v2
from torchvision.io import read_image
import torch

from psutil import virtual_memory, cpu_count
import logging, math
import numpy as np
from tqdm import tqdm
from os import path

from divan.utils.config import *
from divan.utils.log import *
from divan.utils.transformer import *
from divan.utils.utils import *

class DIVANetDataset(Dataset):
    def __init__(self,
                 file_name,
                 label_txt,
                 size=224,
                 channels=None,
                 fix_mean=False,
                 cut_channels=False,
                 random_p=0.8,
                 max_channels=3,
                 ):
        super().__init__()
        assert isinstance(channels, (str, type(None)))
        assert isinstance(size, (int, tuple, list))

        apply_args(self)
        apply_config(self, __file__)

        self.transforms[0].size = (self.size, self.size) if isinstance(self.size, int) else self.size
        self.transforms[-1].size = getattr(torch, self.image_dtype)
        self.transforms[-1].scale = True
        #output_transform = [v2.ToDtype(getattr(torch, self.image_dtype), scale=True)]
        self.transforms = v2.Compose(self.transforms) # + output_transform
        self.RAM = False
        self.fix_mean = True if fix_mean else False
        self.cut_channels = cut_channels if isinstance(channels, str) else False

        self.Dataset_PATH = self.Dataset_PATH + f"{file_name}"
        self._read_data()

    def _read_data(self):
        self._read_txt()
        self._get_class_num()
        self._build_channels_adj()

    def _build_channels_adj(self):
        if self.channels != None:
            self.channels_adj = torch.Tensor(sorted([self.RGB_index.index(c) + 1 for c in self.channels]))
        else:
            self.channels_adj = self.randint((self.__len__(), self.max_channels), self.random_p)

        self.adj_use_idx = len(self.channels_adj.shape) == 2

    def _read_txt(self):
        with open(path.join(self.Dataset_PATH, self.label_txt), 'r') as f:
            lines = list(map(self._split_txt, f.readlines()))
        self.img_label = np.array(lines)

    def _get_class_num(self):
        self.class_num = len(np.unique(self.img_label[:, 1]))

    def _get_image_and_label(self, idx):
        img_path, label = self.img_label[idx]

        channels_idx = self.channels_adj[idx] if self.adj_use_idx else self.channels_adj
        if channels_idx.sum() == 0:
            channels_idx[torch.randint(self.max_channels, (1,))[0]] = 1

        channels_adj = torch.argwhere(channels_idx < 1).flatten()
        _channels = torch.argwhere(channels_idx >= 1).flatten()

        img = read_image(img_path)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = self.transforms(img)

        if not self.cut_channels:
            if len(channels_adj) > 0:
                _fix = torch.mean(img[_channels].float(), dim=0).byte() if self.fix_mean else 0
                img[channels_adj] = _fix

        else:
            img = img[_channels]

        return img, int(label)

    def _updata_img_label(self, img_label):
        self.img_label = img_label
        self.RAM = True if isinstance(self.img_label[0][0], torch.Tensor) else False

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        return self._get_image_and_label(idx) if not self.RAM else self.img_label[idx]

    def _split_txt(self, line):
        line = line.replace('\n','').split(" ")
        line[0] = path.join(self.Dataset_PATH, line[0])
        return line
    @staticmethod
    def randint(shape, p=0.8):
        return (torch.rand(shape) < p).int()


class Dataset_Manager:
    def __init__(self, dataset_path,
                 label_path=["train.txt", "val.txt", "test.txt"],
                 channels='RGB',
                 size=224,
                 batch_size=32,
                 pin_memory=False,
                 shuffle=True,
                 silence=False,
                 fix_mean=False,
                 cut_channels=False,
                 random_p=0.8,
                 num_workers=-1,
                 RAM='auto',
                 cutmix_p=1,
                 ncols=90,
                 RAM_lim=0.925,
                 ):
        assert isinstance(label_path, (tuple, list)) and len(label_path) >= 2
        assert isinstance(size, (int, list, tuple))

        apply_args(self)
        apply_config(self, __file__)
        self.mem = virtual_memory()
        self.RAM = False if not RAM else 'auto'
        self.collate_fn_use = True
        self.num_workers = num_workers if num_workers >= 0 else cpu_count(logical=False)
        self._ready()

    def _ready(self):
        self._buildup_datasets()
        self._class_num()
        self.cutmix = v2.CutMix(num_classes=self.class_num)
        self.one_hot = one_hot(num_classes=self.class_num)
        self._setup_cutmix()

        self._ram_check()

        if self.RAM:
            self._pre_loading()
        else:
            self._build_loader()

    def close_cutmix(self):
        assert hasattr(self, "Data_list")
        self.cutmix_p = 0
        self._build_loader()

    def _buildup_datasets(self):
        _channels = self.channels if isinstance(self.channels,
                                                (tuple, list)) else [self.channels] * len(self.label_path)

        _fix_mean = self.fix_mean if isinstance(self.fix_mean,
                                                (tuple, list)) else [self.fix_mean] * len(self.label_path)

        _cut_channels = self.cut_channels if isinstance(self.cut_channels,
                                                        (tuple, list)) else [self.cut_channels] * len(self.label_path)

        idx_ls = list(range(len(self.data_name)))
        idx_ls[-1] = -1

        for idx, name in zip(idx_ls, self.data_name):
            _dataset = DIVANetDataset(self.dataset_path, self.label_path[idx],
                                      size=self.size,
                                      random_p=self.random_p,
                                      channels=_channels[idx],
                                      fix_mean=_fix_mean[idx],
                                      cut_channels=_cut_channels[idx],
                                      )
            setattr(self, f"{name}_Data", _dataset)

        self.Data_list = [self.train_Data, self.val_Data, self.test_Data]
        self._build_loader()

    def _build_loader(self):
        for idx, (_dataset, name) in enumerate(zip(self.Data_list, self.data_name)):
            collate_fn = self._collate_eval if (idx > 0 or self.cutmix_p<=0) else self._collate_train
            num_workers = min(0 if self.RAM else self.num_workers, 10)
            loader = DataLoader(_dataset,
                                batch_size=self.batch_size,
                                pin_memory=self.pin_memory,
                                shuffle=self.shuffle,
                                num_workers=num_workers,
                                collate_fn=collate_fn
                                )
            setattr(self, f"{name}_loader", loader)

    def _setup_cutmix(self):
        p = [self.cutmix_p, 1-self.cutmix_p]
        if self.cutmix_p > 0:
            self.out_train = v2.RandomChoice([self.cutmix, self.one_hot], p=p)
        else:
            self.out_train = self.one_hot
        self.out_eval = self.one_hot

    def _collate_train(self, batch):
        return self.out_train(*default_collate(batch))

    def _collate_eval(self, batch):
        return self.out_eval(*default_collate(batch))

    def _ram_check(self):
        mem_lim = (self.RAM_lim * self.mem.total) / 1024 ** 3
        system_table = logging_table(self.system_info,
                                    [self.num_workers, round(mem_lim, 1)],
                                    it=['', 'G'],
                                    table_name='System_Info')

        if self.RAM == 'auto':
            logging.info(system_table)
            logging.debug(f"{self.block_name}: RAM auto Checking...")

            used_mem = self.mem.used / 1024 ** 3
            usage_size = self._get_img_size()

            ram_table = logging_table(self.ram_info,
                                      [mem_lim, used_mem, usage_size+used_mem],
                                      table_name='RAM_Info',
                                      it='G')

            self.RAM = True if math.ceil(mem_lim) >= usage_size + used_mem else False
            logging.info(ram_table)

        logging.info(f"{self.block_name}: RAM Temporary - {self.RAM}")

    def _get_img_size(self):
        assert hasattr(self, "Data_list")
        img_num = sum([len(i) for i in self.Data_list])
        H, W = (self.size, self.size) if isinstance(self.size, int) else self.size[:2]
        itemsize = getattr(torch, self.image_dtype).itemsize

        self.ram_size = img_num * (3 * itemsize * H * W) / 1024 ** 3
        return self.ram_size

    def _class_num(self):
        assert hasattr(self, "Data_list")
        self.class_num =  max([i.class_num for i in self.Data_list])

    def _pre_loading(self, show=True):
        assert hasattr(self, 'Data_list')
        for idx, dataset in enumerate(self.Data_list):
            self._pre_loading_step(dataset, show)
        self.shuffle = False
        self._build_loader()

    def _pre_loading_step(self, _dataset, show):
        set_sharing_strategy('file_system')
        batch_size = 128
        img_ls, label_ls = [], []
        loader = DataLoader(_dataset,
                            batch_size=batch_size,
                            shuffle=self.shuffle,
                            num_workers=min(self.num_workers, round(len(_dataset)//batch_size)),
                            )

        desc = f"{self.block_name}: Dataset pre-loading"
        pbar = tqdm(loader, desc=desc, ncols=self.ncols) if show else loader
        for img, label in pbar:
            img_ls.append(img)
            label_ls.append(label)

        set_sharing_strategy('file_descriptor')
        imgs = torch.cat(img_ls, dim=0)
        labels = torch.cat(label_ls, dim=0)

        img_label = list(zip(imgs, labels))
        _dataset._updata_img_label(img_label)


if __name__ == '__main__':
    DIVANetDataset('dataset', 'train.txt')