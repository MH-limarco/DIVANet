from divan.module.utils import *
from divan.module.conv import *

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ChannelAttention", "SpatialAttention", "CBAM", "PSA"]

class ChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction=16,
                 layer='Conv2d',
                 activation='ReLU',
                 pool=['avg', 'max'],
                 kernel_size=3,
                 stride=1,
                 norm_type=2,
                 ):
        super(ChannelAttention, self).__init__()
        assert layer in ['Conv2d', 'Linear']
        for p in pool:
            assert p in ['avg', 'max', 'lp']

        self.flatten = layer == 'Linear'
        self.pool_types = [getattr(F, p + '_pool2d') for p in pool]
        self.norm_type = norm_type

        reduced_channels = max(in_channels // reduction, 1)
        cnn_args = {'kernel_size' : kernel_size, 'padding' : stride} if layer == 'Conv2d' else {}

        ff_layer = getattr(nn, layer)
        ff_act = Activations(act_name=activation)

        _begin_Sequential = [Flatten()] if self.flatten else []
        _Sequential = (
                _begin_Sequential +
                [ff_layer(in_channels, reduced_channels, **cnn_args),
                 ff_act(),
                 ff_layer(reduced_channels, in_channels,  **cnn_args)])
        self.mlp = nn.Sequential(*_Sequential)

    def forward(self, x):
        channel_att_sum = None
        dim = x.size()

        for pool_type in self.pool_types:
            ex_args = {'norm_type': self.norm_type} if pool_type == F.lp_pool2d else {}

            pool_x = pool_type(x, kernel_size=(dim[2], dim[3]), stride=(dim[2], dim[3]), **ex_args,)
            channel_att_raw = self.mlp(pool_x)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw

            else:
                channel_att_sum += channel_att_raw

        scale = F.sigmoid(channel_att_sum)
        if self.flatten:
            scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, activation=False, kernel_size=7):
        super(SpatialAttention, self).__init__()
        in_channels, out_channels = 2, 1
        padding = max((kernel_size - 1) // 2, 1)

        self.pool = ChannelPool()
        self.spatial = Conv(in_channels, out_channels, kernel_size,
                            stride=1,
                            padding=padding,
                            activation=activation)

    def forward(self, x):
        x_pool = self.pool(x)
        x_out = self.spatial(x_pool)
        scale = F.sigmoid(x_out)
        return x * scale

class CBAM(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction=16,
                 layer='Conv2d',
                 activation='SiLU',
                 pool=['avg', 'max'],
                 kernel_size=3,
                 stride=1,
                 norm_type=2,
                 spatial=True
                 ):
        super(CBAM, self).__init__()
        _Sequential = [ChannelAttention(in_channels,
                                         reduction,
                                         layer,
                                         activation,
                                         pool,
                                         kernel_size,
                                         stride,
                                         norm_type)]
        if spatial:
            _Sequential.append(SpatialAttention())

        self.Sequential = nn.Sequential(*_Sequential)

    def forward(self, x):
        x_out = self.Sequential(x)
        return x_out

class QKV(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, activation=False)
        self.proj = Conv(dim, dim, 1, activation=False)
        self.pe = Conv(dim, dim, 3, 1, groups=dim, activation=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x



class PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert (c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.attn = QKV(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1),
            Conv(self.c * 2, self.c, 1, activation=False)
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

if __name__ == '__main__':
    img = torch.randn((4, 128, 224, 224))
    print(PSA(128,128)(img).shape)