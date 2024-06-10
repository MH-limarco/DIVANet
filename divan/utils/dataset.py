from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision.transforms import v2
from torchvision.io import read_image
import torch

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from psutil import virtual_memory
from tqdm import tqdm
from os import path, walk
import logging, asyncio
import numpy as np

from divan.utils import *



class dataset_Manager(Dataset):
    def __init__(self, data_name, label_txt,
                 ):
        super().__init__()
        utils.apply_args(self)
        config.apply_config(self, __file__)

        output_transform = [v2.ToDtype(getattr(torch, self.image_dtype), scale=True)]
        self.transforms = v2.Compose(self.transforms + output_transform)
        self.RAM = False
        self.Dataset_PATH = self.Dataset_PATH + f"{self.data_name}"
        self.read()

    def read(self):
        self._read_txt()

    def _read_txt(self):
        with open(path.join(self.Dataset_PATH, self.label_txt), 'r') as f:
            lines = list(map(self._split_txt, f.readlines()))
        self.img_label = np.array(lines)

    def _RAM_check(self):
        pass
    def _get_class_num(self):
        self.class_num = len(np.unique(self.img_label[:, 1]))

    def _get_image_and_label(self, idx):
        img_path, label = self.img_label[idx]

        img = read_image(img_path)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = self.transforms(img)
        return img, label

    def _pre_loading(self):
        pass

    async def _pre_loading_async(self, desc):
        pass

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        return self._get_image_and_label(idx) if not self.RAM else None


    def _split_txt(self, line):
        line = line.replace('\n','').split(" ")
        line[0] = path.join(self.Dataset_PATH, line[0])
        return line



class DIVANetDataset:
    pass

if __name__ == '__main__':
    dataset_Manager('dataset', 'train.txt')





