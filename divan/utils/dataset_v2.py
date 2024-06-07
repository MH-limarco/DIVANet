from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.nn.functional import one_hot
import torch

from concurrent.futures import ThreadPoolExecutor
from psutil import virtual_memory
from tqdm import tqdm
from os import path, walk
import logging

from divan.utils.log import *

if __name__ == '__main__':
    from transformer import *
else:
    from .transformer import *

block_name = 'Dataset_Manager'
ram_table_col = ['Available_mem', 'Usage_mem', 'Intended_mem']
datset_table_col = ['num_workers', 'RAM_lim']
RGB_index = ['R', 'G', 'B']

image_dtype = torch.uint8

_transforms = [v2.RandomResizedCrop(size=(224, 224), antialias=True),
               v2.RandomHorizontalFlip(p=0.5),
               v2.ToDtype(torch.float32, scale=True),
               v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
               v2.ToDtype(image_dtype, scale=True)]

_transforms = v2.Compose(_transforms)


class base_Dataset_v2(Dataset):
    def __init__(self,
                 dataset_path,
                 label_path,
                 transform=None,
                 channels='RGB',
                 channels_mode='smooth',
                 silence=False,
                 image_path='images',
                 num_workers=-1,
                 RAM='auto',
                 RAM_lim=0.9,
                 ncols=90
                 ):
        assert RAM_lim <= 1
        assert channels_mode in ['smooth', 'hard', 'auto']
        channels = list(channels) if isinstance(channels, str) else channels

        self.dataset_path = dataset_path
        self.label_path = label_path
        self.image_path = image_path
        self.silence = silence
        self.ncols = ncols

        self.channels_mode = channels_mode == 'hard'
        if channels_mode != 'auto':
            channels = [RGB_index.index(c) for c in channels]
            channels = channels if self.channels_mode else sorted(channels)
            self.channels_adj = channels != list(range(len(RGB_index)))

            mask = torch.ones(len(RGB_index), dtype=torch.bool)
            mask[channels] = False
            self.channels = channels if self.channels_mode else mask
        else:
            self.channels_adj = False

        self.num_workers = num_workers
        self.RAM = RAM if not RAM else 'auto'
        self.RAM_lim = RAM_lim

        self.transform = _transforms if transform == None else transform

        self.mem = virtual_memory()
        self._ready_step()

    def _ready_step(self):
        self._read_txt()
        self._get_class_num()
        self._RAM_check()
        if self.RAM:
            self._pre_loading()

    def _read_txt(self):
        with open(path.join(self.dataset_path, self.label_path), 'r') as f:
            lines = list(map(self._split_txt, f.readlines()))
        self.img_label = lines

    def _RAM_check(self):
        mem_lim = (self.RAM_lim * self.mem.total) / (1024 ** 3)
        logging_table(datset_table_col, [self.num_workers, mem_lim], table_name='Datset-setting')
        if self.RAM == 'auto':
            if not self.silence:
                logging.debug(f"{block_name}: {self.label_path} RAM Temporary - {self.RAM}")
                logging.debug(f"{block_name}: {self.label_path} RAM auto Checking...")
            size = 0
            for root, dirs, files in walk(path.join(self.dataset_path, self.image_path)):
                size += sum([path.getsize(path.join(root, name)) for name in files])
            file_size = ((size * image_dtype.itemsize * 8) / 1024 ** 3)

            use_mem = self.mem.used / 1024 ** 3
            ram_table_value = [mem_lim, use_mem, file_size]
            logging_table(ram_table_col, ram_table_value, table_name='RAM-Check-Table', it='G')

            self.RAM = True if mem_lim < file_size + self.mem.used else False

        if not self.silence:
            logging.info(f"{block_name}: {self.label_path} RAM Temporary - {self.RAM}")

    def _get_class_num(self):
        self.class_num = len(np.unique(np.array(self.img_label)[:, 1]))

    def _get_image_and_label(self, idx):
        img_path, label = self.img_label[idx]
        img = read_image(path.join(self.dataset_path, img_path))
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        if self.transform:
            img = self.transform(img)

        if self.channels_adj:
            if self.channels_cut:
                img = img[self.channels, :, :]
            else:
                img[self.channels] = 0

        return img, int(label)

    def _pre_loading(self):
        desc = f"{block_name}: Dataset pre-loading"
        if self.num_workers <= 1:
            pbar = range(self.__len__()) if self.silence else tqdm(range(self.__len__()),
                                                                   desc=desc,
                                                                   ncols=self.ncols)
            for i in pbar:
                self.img_label[i] = self._get_image_and_label(i)

        else:
            with ThreadPoolExecutor() as executor:
                mp_map = executor.map(self._get_image_and_label,
                                      range(self.__len__()))

                self.img_label = list(mp_map if self.silence else tqdm(mp_map,
                                                                   total=self.__len__(),
                                                                   desc= desc,
                                                                   ncols=self.ncols
                                                                   ))

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        return self.img_label[idx] if self.RAM else self._get_image_and_label(idx)

    @staticmethod
    def _split_txt(line):
        return line.replace('\n','').split(" ")

class MiniImageNetDataset:
    def __init__(self,
                 dataset_path,
                 label_path,
                 transform=transforms,
                 channels='RGB',
                 channels_mode='smooth',
                 batch_size=32,
                 shuffle=False,
                 pin_memory=False,
                 silence=False,
                 cutmix_p=0,
                 image_path='images',
                 num_workers=-1,
                 RAM='auto',
                 RAM_lim=0.9,
                 ncols=90
                 ):
        assert RAM_lim <= 1
        assert 0 <= cutmix_p <= 1
        assert channels_mode in ['smooth', 'hard', 'auto']

        self.dataset_path = dataset_path
        self.label_path = label_path
        self.image_path = image_path
        self.silence = silence
        self.ncols = ncols

        self.num_workers = num_workers
        self.RAM = RAM if not RAM else 'auto'
        self.RAM_lim = RAM_lim

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.transform = transform
        self.cutmix_p = cutmix_p
        self.shuffle = True if self.cutmix_p > 0 else shuffle

        self.dataset = base_Dataset_v2(dataset_path,
                                       label_path,
                                       transform,
                                       channels,
                                       channels_mode,
                                       silence,
                                       image_path,
                                       num_workers,
                                       RAM,
                                       RAM_lim,
                                       ncols)

        self.class_num = self.dataset.class_num
        self.cutmix = v2.CutMix(num_classes=self.class_num)
        self.one_hot = one_hot(num_classes=self.class_num)
        self.build_loader()

    def setup_cutmix(self):
        p = [self.cutmix_p, 1-self.cutmix_p]
        self._out = v2.RandomChoice([self.cutmix, self.one_hot], p=p)
        self._out2 = v2.RandomChoice([self.one_hot])

    def build_loader(self):
        self.setup_cutmix()
        self.DataLoader = DataLoader(self.dataset,
                                     self.batch_size,
                                     shuffle=self.shuffle,
                                     pin_memory=self.pin_memory,
                                     num_workers=min(self.num_workers, 4 if not self.RAM else 0),
                                     collate_fn=self.collate_fn
                                     )

    def collate_fn(self, batch):
        return self._out(*default_collate(batch))

    def close_cutmix(self):
        self.cutmix_p = 0
        self.build_loader()

if __name__ == '__main__':
    FORMAT = '%(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt='%Y-%m-%d %H:%M')

    DataManager = MiniImageNetDataset('../../dataset/dataset', 'train.txt', RAM=False, pin_memory=True, silence=False)
    DataManager.build_loader()
    for img, _label in tqdm(DataManager.DataLoader):
        pass
    print(torch.max(_label), torch.sum(_label,axis=1), img.shape)

    DataManager2 = MiniImageNetDataset('../../dataset/dataset', 'train.txt', RAM='auto', pin_memory=True, silence=False)
    DataManager2.build_loader()
    for img, _label in tqdm(DataManager2.DataLoader):
        pass
    print(torch.max(_label), torch.sum(_label,axis=1), img.shape, _label)




