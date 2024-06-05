from torch import nn
import torch

from src.module.utils import *
from src.module.kan_convs import *

try:
    from DCNv4.modules.dcnv4 import DCNv4 as dcnv4
except ImportError as e:
    pass

__all__ = ["Conv", "DCNv4", "MLPLayer", "Concat"]

class Conv(nn.Module):
    default_act = Activations('SiLU')(inplace=True)
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 activation=True,
                 bn=True,
                 bias=False
                 ):
        super(Conv, self).__init__()
        self.out_channels = out_planes
        self.activation = activation

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=autopad(kernel_size, padding, dilation), dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else nn.Identity()
        self.activation = self.default_act if activation else activation if isinstance(activation, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class DCNv4(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 activation=True,
                 bn=True,
                 ):
        super().__init__()

        if in_planes != out_planes:
            self.stem_conv = Conv(in_planes, out_planes, kernel_size=1)

        kernel_size = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        self._conv = dcnv4(out_planes,
                           kernel_size=kernel_size,
                           stride=stride,
                           pad=autopad(kernel_size, padding, dilation),
                           group=groups,
                           dilation=dilation)

        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else nn.Identity()
        self.activation = Conv.default_act if activation is True else activation if isinstance(activation, nn.Module) else nn.Identity()

    def forward(self, x):
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)

        x = self._conv(x, (x.size(2), x.size(3)))
        x = self.activation(self.bn(x))
        return x

class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 mlp_fc2_bias=False,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = Activations(act_layer)(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_fc2_bias)
        self.drop = nn.Dropout(drop)


    def forward(self, x, shape):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Concat(nn.Module):
    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)

if __name__ == "__main__":

    model = dcnv4(4, 16, 3).cuda()
    input_rgb = torch.randn(4, 4, 224, 224).cuda()

    print(input_rgb.shape)
    print(model(input_rgb).shape)