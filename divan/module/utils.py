from torch import nn
from torch.nn import functional as F
import torch, math

__all__ = ["make_divisible", "Flatten", "ChannelPool", "autopad", "Activations",
           "inlayer_resize", "fclayer_resize"]

def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

####################################   att   ##############################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

#################################   cnn layer setting  #####################

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


####################### Activations ##########################

class SwiGLU(nn.Module):
    __constants__ = ['inplace']
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        assert x.shape[-1] % 2 == 0
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def Activations(act_name):
    if act_name == 'SwiGLU':
        return SwiGLU
    else:
        return getattr(nn, act_name)

def inlayer_resize(model, in_channels=3):
    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Conv2d):
            new_layer = nn.Conv2d(in_channels, layer.out_channels,
                                  kernel_size=layer.kernel_size,
                                  stride=layer.stride,
                                  padding=layer.padding,
                                  bias=False if layer is None else True)

            name_parts = name.split('.')
            sub_module = model
            for part in name_parts[:-1]:
                sub_module = getattr(sub_module, part)
            setattr(sub_module, name_parts[-1], new_layer)
            return model

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1,4) to box2(n,4).
    Args:
        box1: (tensor) pred boxes
        box2: (tensor) target boxes
        xywh: (bool) if True, boxes are (x,y,w,h), else (x1,y1,x2,y2)
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou

def fclayer_resize(model, num_class=1000):
    for name, layer in reversed(list(model.named_modules())):
        if isinstance(layer, nn.Linear):
            new_layer = nn.Linear(layer.in_features, out_features=num_class, bias=False if layer is None else True) #

            name_parts = name.split('.')
            sub_module = model
            for part in name_parts[:-1]:
                sub_module = getattr(sub_module, part)
            setattr(sub_module, name_parts[-1], new_layer)
            return model

if __name__ == '__main__':
    from divan.module.backbone import vision_backbone
    model = vision_backbone('resnet34')
    print(fclayer_resize(model, 50))



