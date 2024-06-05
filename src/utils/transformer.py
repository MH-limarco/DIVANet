import math, random
import numpy as np

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.v2._transform import Transform

class Compose:
    """Class for composing multiple image transformations."""
    def __init__(self, transforms):
        """Initializes the Compose object with a list of transforms."""
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """Applies a series of transformations to input data."""
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Appends a new transform to the existing list of transforms."""
        self.transforms.append(transform)

    def insert(self, index, transform):
        """Inserts a new transform to the existing list of transforms."""
        self.transforms.insert(index, transform)

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        """Retrieve a specific transform or a set of transforms using indexing."""
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        index = [index] if isinstance(index, int) else index
        return Compose([self.transforms[i] for i in index])

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        """Retrieve a specific transform or a set of transforms using indexing."""
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(
                value, list
            ), f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        """Converts the list of transforms to a standard Python list."""
        return self.transforms

    def __repr__(self):
        """Returns a string representation of the object."""
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"

class BaseMixTransform:
    def __init__(self, dataset, pre_transform=None, p=0.0):
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        """Applies pre-processing transforms and mixup/mosaic transforms to labels data."""
        if random.uniform(0, 1) > self.p:
            return self.dataset._nor_transform(labels)

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic or MixUp
        mix_labels = [self.dataset._get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # Update cls and texts
        labels['labels'] = self._update_label_text(labels)

        # Mosaic or MixUp
        fin_labels = self._mix_transform(labels)


        labels.pop("mix_labels", None)
        return fin_labels

    def _mix_transform(self, labels):
        """Applies MixUp or Mosaic augmentation to the label dictionary."""
        raise NotImplementedError

    def get_indexes(self):
        """Gets a list of shuffled indexes for mosaic augmentation."""
        raise NotImplementedError

    def _update_label_text(self, labels):
        """Update label text."""
        mix_texts = [labels["label"]] + [x["label"] for x in labels["mix_labels"]]
        return mix_texts

class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation.

    This class performs mosaic augmentation by combining multiple 4 images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int, optional): Image size (height and width) after mosaic pipeline of a single image. Default to 640.
        p (float, optional): Probability of applying the mosaic augmentation. Must be in the range 0-1. Default to 1.0.
        n (int, optional): The grid size, either 4 (for 2x2) or 9 (for 3x3).
    """

    def __init__(self, dataset, imgsz, p=1.0):
        """Initializes the object with a dataset, image size, probability, and border."""
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        super().__init__(dataset=dataset, p=p)
        self.dataset = dataset
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = 4

        self.ToTensor = transforms.ToTensor()
        self.class_num = dataset.class_num

    def get_indexes(self):
        return random.choices(list(range(len(self.dataset)-1)), k=self.n - 1)

    def _mix_transform(self, data):
        """Apply mixup transformation to the input image and labels."""
        assert len(data.get("mix_labels", [])), "There are no other images for mosaic augment."
        return self.dictLabel_2_onehot(self._mosaic4(data))

    def _mosaic4(self, data):
        """Create a 2x2 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = data if i == 0 else data["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 0, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)

                xmin, ymin = x1a, y1a
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                xmax, ymax = x2a, y2a

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            mosaic_labels.append((y2a - y1a)*(x2a - x1a))

        final_labels = self._cat_labels(data, mosaic_labels)
        final_labels["img"] = self.ToTensor(img4[xmin:ymax, ymin:xmax])
        return final_labels

    def _cat_labels(self, data, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        tol_ = sum(mosaic_labels)
        mosaic_labels = [i/tol_ for i in mosaic_labels]

        if len(mosaic_labels) == 0:
            return {}

        # Final labels
        final_labels = {"label": {idx:x for idx, x in zip(data['labels'], mosaic_labels)}}
        return final_labels

    def dictLabel_2_onehot(self, data):
        one_hot = torch.zeros((self.class_num))
        for idx, label in data['label'].items():
            one_hot[idx] = label
        data['label'] = one_hot
        return data

class one_hot(Transform):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, *inputs):
        img, label = inputs
        label = F.one_hot(label, num_classes=self.num_classes).float()
        return img, label

if __name__ == "__main__":
    from dataset import *

    #train_dataset = MiniImageNetDataset('../dataset/dataset', 'train.txt')
    #a = Compose([Mosaic(train_dataset, 224)])
    #print(a)
