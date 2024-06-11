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
    dataset = Dataset_Manager('dataset', batch_size=64, cutmix_p=0.0, channels='RGB', RAM=False, shuffle=True)
    for img, label, c_idx in tqdm(dataset.train_loader):
        pass

    dataset.close_cutmix()

    for img2, label2, c_idx2 in tqdm(dataset.train_loader):
        pass

    for img, label, c_idx in tqdm(dataset.val_loader):
        pass

    for img, label, c_idx in tqdm(dataset.test_loader):
        pass
    #dataset_load = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=40)
    #for img, label, c_idx in tqdm(dataset_load):
    #    pass
