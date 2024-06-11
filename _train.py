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
    dataset = Dataset_Manager('dataset', channels=None, RAM=False)
    for i in tqdm(dataset.train_loader):
        pass
    #dataset_load = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=40)
    #for img, label, c_idx in tqdm(dataset_load):
    #    pass
