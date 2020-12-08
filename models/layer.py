import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from .inplace_abn import InPlaceABN, ABN
__all__ = [
    'conv3x3x3', 'ResBlock', 'Bottleneck', 'Basic_block_inplace', 'BasicConvBN3d','ResBlockGN','Conv3d_ABN','Bottleneck_inplace']

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class Mish(nn.Module):
    def __init__(seelf):
        super().__init__()
        
    def forward(self, x):
        x  = x*(torch.tanh(F.softplus(x)))
        return x

class Conv3d_ABN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.abn = self.abnblock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x
    @classmethod
    def define_abn(cls, abn):
        cls.abnblock = abn
        
class Conv3d_Mish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


        
class Bottleneck_ABN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, stride=1, downsample=None):
        super().__init__()
        planes = inplanes//self.expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self.abnblock(inplanes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = self.abnblock(planes)
        self.conv3 = nn.Conv3d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = self.abnblock(planes)
        self.downsample = downsample
        self.stride = stride
        
    @classmethod
    def define_abn(cls, abn):
        cls.abnblock = abn

    def forward(self, x):
        out = x.clone()

        out = self.bn1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            x = self.downsample(x)


        out += x

        return out
    
class Bottleneck_Mish(nn.Module):
    expansion = 4

    def __init__(self, inplanes, stride=1, downsample=None):
        super().__init__()
        planes = inplanes//self.expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu = Mish()
        

    def forward(self, x):
        out = x.clone()

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            x = self.downsample(x)


        out += x

        return out
    

class BasicConvTranspose3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConvTranspose3d, self).__init__()
        self.convTranspose = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                                padding=padding)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.convTranspose(x)
        x = self.relu(x)
        return x


class BasicConvBN3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConvBN3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicConvTransposeBN3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConvTransposeBN3d, self).__init__()
        self.convTranspose = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                                padding=padding)

        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.convTranspose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias)

class ResBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock2, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResBlockGN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlockGN, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(4, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.GroupNorm(4, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Basic_block_inplace(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, abn=InPlaceABN):
        super(Basic_block_inplace, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, bias = True)
        self.bn1 = abn(planes)
        self.conv2 = conv3x3x3(planes, planes,bias = True)
        self.bn2 = abn(planes)
        self.downsample = downsample
        self.stride = stride
        #self.conv2.weight.data = self.conv2.weight.data*0.1
        #self.conv2.bias.data.fill_(0)

    def forward(self, x):
        xcopy = x.clone()
        out = self.bn1(xcopy)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        return out + x




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


