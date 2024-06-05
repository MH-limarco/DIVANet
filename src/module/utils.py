from torch import nn
from torch.nn import functional as F
import torch, math

__all__ = ["make_divisible", "Flatten", "ChannelPool", "autopad", "Activations"]

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


