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



