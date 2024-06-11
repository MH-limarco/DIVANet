import numpy as np

from divan import *
from divan.utils.dataset import DIVANetDataset
from divan.check.check_file import check_file
from tqdm import tqdm
import time
import torch.multiprocessing

if __name__ == "__main__":
    FORMAT = '%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=FORMAT)
    start = time.monotonic()
    dataset = Dataset_Manager('dataset', num_workers=20, batch_size=64, channels='RGB', RAM=True, shuffle=True)
    for img, label, c_idx in tqdm(dataset.train_loader):
        print(label.max(axis=0), label.max(axis=1))
        break

    dataset.close_cutmix()

    for img2, label2, c_idx2 in tqdm(dataset.train_loader):
        print(label2.max(axis=0), label2.max(axis=1))
        break

    for img, label, c_idx in tqdm(dataset.val_loader):
        pass

    for img, label, c_idx in tqdm(dataset.test_loader):
        pass
    #dataset_load = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=40)
    #for img, label, c_idx in tqdm(dataset_load):
    #    pass
