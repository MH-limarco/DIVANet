from divan import *
from divan.utils.dataset import dataset_Manager
from divan.check.check_file import check_file
from tqdm import tqdm

if __name__ == "__main__":
    dataset = dataset_Manager('dataset', 'train.txt')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    for img, label in tqdm(dataloader):
        pass
