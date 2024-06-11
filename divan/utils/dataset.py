from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision.transforms import v2
from torchvision.io import read_image
import torch

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from psutil import virtual_memory, cpu_count
from tqdm import tqdm
from os import path, walk
import logging, asyncio
import numpy as np

from divan.utils import *

#dataset_Manager

class DIVANetDataset(Dataset):
    def __init__(self,
                 data_name,
                 label_txt,
                 channels=None,
                 channels_mode='smooth',
                 max_channels=3,
                 random_p=0.7,
                 ):
        super().__init__()
        assert isinstance(channels, (str, type(None)))
        utils.apply_args(self)
        config.apply_config(self, __file__)
        self.RAM = False

        output_transform = [v2.ToDtype(getattr(torch, self.image_dtype), scale=True)]
        self.transforms = v2.Compose(self.transforms + output_transform)

        self.Dataset_PATH = self.Dataset_PATH + f"{self.data_name}"
        self._read_data()

    def _read_data(self):
        self._read_txt()
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

        if self.adj_use_idx:
            channels_adj = torch.argwhere(channels_idx < 1).flatten()
        else:
            channels_adj = torch.argwhere(channels_idx < 1)

        img = read_image(img_path)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = self.transforms(img)

        img[channels_adj] = 0

        return img, int(label), channels_idx.bool()

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
    def __init__(self,
                 dataset_path,
                 label_path,
                 channels='RGB',
                 channels_mode='smooth',
                 batch_size=32,
                 shuffle=True,
                 silence=False,
                 cutmix_p=0,
                 ):
        pass

if __name__ == '__main__':
    DIVANetDataset('dataset', 'train.txt')





