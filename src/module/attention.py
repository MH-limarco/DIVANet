from src.module.utils import *
from src.module.conv import *

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ChannelAttention", "SpatialAttention", "CBAM"]

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
        ff_act = Activations(act_name=activation)()

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
        self.spatial = Conv(in_channels, out_channels, kernel_size, stride=1, padding=padding, activation=activation)

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

#if __name__ == '__main__':
#    class FlexibleInputCNN(nn.Module):
#        def __init__(self, input_channels):
#            super(FlexibleInputCNN, self).__init__()
#            self.channel_attention = CBAM(in_channels=input_channels)
#            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
#            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
#            self.fc1 = nn.Linear(64 * 7 * 7, 128)
#            self.fc2 = nn.Linear(128, 10)

#        def forward(self, x):
#            # 应用通道注意力
#            x = self.channel_attention(x)
#
#            x = F.relu(self.conv1(x))
#            x = F.relu(self.conv2(x))
#            x = self.adaptive_pool(x)

#            x = x.view(x.size(0), -1)  # 展平
#            x = F.relu(self.fc1(x))
#            x = self.fc2(x)

#            return x
#    def create_model_for_input_channels(input_channels):
#        return FlexibleInputCNN(input_channels=input_channels)

#    input_rgb = torch.randn(4, 3, 224, 224)
#    input_rg = torch.randn(4, 2, 32, 32)
#    input_gray = torch.randn(4, 1, 28, 28)

#    model_rgb = create_model_for_input_channels(3)
#    model_rg = create_model_for_input_channels(2)
#    model_gray = create_model_for_input_channels(1)

#    output_rgb = model_rgb(input_rgb)
#    output_rg = model_rg(input_rg)
#    output_gray = model_gray(input_gray)

#    print("Output shape for RGB input:", output_rgb.shape)
#    print("Output shape for RG input:", output_rg.shape)
#    print("Output shape for grayscale input:", output_gray.shape)