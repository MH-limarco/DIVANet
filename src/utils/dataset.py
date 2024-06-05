import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image


class base_Dataset(Dataset):
    def __init__(self, dataset_path, label_txt,
                 size=224,
                 transform=[],
                 mosaic_p=0,
                 ):
        self.dataset_path = dataset_path
        self.label_txt = label_txt

        if transforms.Resize((size, size)) not in transform:
            transform.append(transforms.Resize((size, size)))
        self.transform = transform
        self.mosaic_p = mosaic_p
        self.size = size


        self.label = self._read_txt()
        self.class_num = self._get_class_num()
        self.eyes = torch.eye(self.class_num)

        self._mosaic = Mosaic(self, self.size, p=self.mosaic_p)
        self._build_transforms()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self._get_image_and_label(idx)
        return self._compose(data)

    def close_mosaic(self):
        self.mosaic_p = 0
        self._build_transforms()

    def _read_txt(self):
        with open(os.path.join(self.dataset_path, self.label_txt), 'r') as f:
            lines = np.array(list(map(self._split_txt, f.readlines())))
        return lines

    def _get_class_num(self):
        return len(np.unique(self.label[:, 1]))

    def _get_image_and_label(self, idx):
        img_path, label = self.label[idx]
        img = read_image(os.path.join(self.dataset_path, img_path))
        if img.shape[0]==1:
            img = img.repeat(3, 1, 1)

        return {'img':img.permute(1, 2, 0), 'label':int(label), 'resized_shape':(self.size, self.size)}

    def _build_transforms(self):
        if self.mosaic_p > 0:
            self.mosaic_use = True
        else:
            self.mosaic_use = False

        if self.transform is not None:
            self._transforms = transforms.Compose(self.transform)

    def _compose(self, data):
        if self.mosaic_use:
            data = self._mosaic(data)
        else:
            data = self._nor_transform(data)

        if isinstance(self.transform , list) and len(self.transform) > 0:
            data['img'] = self._transforms(data['img'])
        return data

    def _nor_transform(self, data):
        data['img'] = data['img'].permute(2, 0, 1)
        data['label'] = self.eyes[data['label']]
        return data

    @staticmethod
    def _split_txt(line):
        return line.replace('\n','').split(" ")

class MiniImageNetDataset(base_Dataset):
    pass


if __name__ == '__main__':
    from src.utils.transformer import *
    train_dataset = MiniImageNetDataset('../../dataset/dataset', 'train.txt', mosaic_p=1)
    test_dataset = MiniImageNetDataset('../../dataset/dataset', 'test.txt')
    val_dataset = MiniImageNetDataset('../../dataset/dataset', 'val.txt')

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=12)

    from tqdm import tqdm
    for data in tqdm(train_loader):
        pass




