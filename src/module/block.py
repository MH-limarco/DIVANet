import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.utils import *
from src.module.conv import *

__all__ = ["C1", "C2", "C2f", "C3"]

class Pool_Conv(nn.Module):
    def __init__(self,
                 c1,
                 reduction=16,
                 layer='Conv2d',
                 activation='ReLU',
                 pool=['avg', 'max'],
                 kernel_size=3,
                 stride=1,
                 norm_type=2,
                 ):

        assert layer in ['Conv2d', 'Linear']
        for p in pool:
            assert p in ['avg', 'max', 'lp']

        self.flatten = layer == 'Linear'
        self.pool_types = [getattr(F, p + '_pool2d') for p in pool]
        self.norm_type = norm_type

        reduced_channels = max(c1 // reduction, 1)
        cnn_args = {'kernel_size' : kernel_size, 'padding' : stride} if layer == 'Conv2d' else {}

        ff_layer = getattr(nn, layer)
        ff_act = Activations(act_name=activation)

        _begin_Sequential = [Flatten()] if self.flatten else []
        _Sequential = (
                _begin_Sequential +
                [ff_layer(c1, reduced_channels, **cnn_args),
                 ff_act(),
                 ff_layer(reduced_channels, c1, **cnn_args)])

        self.mlp = nn.Sequential(*_Sequential)



class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    def __init__(self, c1, c2, n=1, _conv=Conv, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, _conv, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, _conv=Conv, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, _conv, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, _conv=Conv, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, _conv, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, _conv, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        try:
            self.cv2 = _conv(c_, c2, k[1], 1, groups=g)
        except:
            self.cv2 = _conv(c_, c2, k[1])

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

#class Bottleneck_InternImage(nn.Module)

if __name__ == '__main__':

    model = C2(3, 16)
    input_rgb = torch.randn(4, 3, 224, 224)

    print(input_rgb.shape)
    print(model(input_rgb).shape)