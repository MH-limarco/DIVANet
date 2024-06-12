import torch
from torchvision.transforms import v2

__all__ = ["Transforms"]

Transforms = [v2.RandomResizedCrop(size=(224, 224), antialias=True),
              v2.RandomHorizontalFlip(p=0.5),
              v2.ToDtype(torch.float32, scale=True),
              v2.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
              v2.ToDtype(torch.uint8, scale=True)
              ]

##v2.ToDtype(image_dtype, scale=True)