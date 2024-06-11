import numpy as np

from divan import *
from divan.utils.dataset import DIVANetDataset
from divan.check.check_file import check_file
from tqdm import tqdm
import time
import torch.multiprocessing

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')

    start = time.monotonic()
    dataset = DIVANetDataset('dataset', 'train.txt',channels=None)
    dataset_load = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=40)
    for img, label, c_idx in tqdm(dataset_load):
        print(c_idx)
