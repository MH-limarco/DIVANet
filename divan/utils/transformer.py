import torch.nn.functional as F
from torchvision.transforms.v2._transform import Transform

__all__ = ["one_hot"]

class one_hot(Transform):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, *inputs):
        img, label = inputs
        label = F.one_hot(label, num_classes=self.num_classes).float()
        return img, label