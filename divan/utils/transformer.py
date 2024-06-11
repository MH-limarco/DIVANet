from torchvision.transforms.v2._transform import Transform
from torchvision.transforms.v2 import RandomChoice
import torch.nn.functional as F
from torch import nn
import torch

__all__ = ["one_hot", "randomChoice", "identity"]

class one_hot(Transform):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    def forward(self, *inputs):
        img, label = inputs
        label = F.one_hot(label, num_classes=self.num_classes).float()

        return img, label

class randomChoice(RandomChoice):
    def forward(self, *inputs):
        idx = int(torch.multinomial(torch.tensor(self.p), 1))
        transform = self.transforms[idx]
        long_input = True if len(inputs) >= 3 else False
        if long_input:
            img, label, c_idx = inputs
        else:
            img, label = inputs

        img, label = transform(img, label)

        if long_input:
            return img, label, c_idx
        else:
            return img, label

class identity(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
    def forward(self, *inputs):
        long_input = True if len(inputs) >= 3 else False
        if long_input:
            img, label, c_idx = inputs
        else:
            img, label = inputs

        img, label = self.transform(img, label)

        if long_input:
            return img, label, c_idx
        else:
            return img, label