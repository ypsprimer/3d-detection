import torch
from torch import nn
from torch.nn import functional as F

class Deform_conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Deform_conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.conv_deform = nn.Conv3d(in_channels, 3, kernel_size=5, padding=2)
        self.conv_deform.weight.data.fill_(0)
        self.conv_deform.bias.data.fill_(0)
        self.first = True


    def forward(self, x):
        if self.first:
            shape = x.shape[2:]
            xx, yy, zz = torch.meshgrid([torch.linspace(-1, 1, shape[0]), torch.linspace(-1, 1, shape[1]), torch.linspace(-1,1,shape[2])])
            regular_grid = torch.cat([xx.unsqueeze(3), yy.unsqueeze(3), zz.unsqueeze(3)], dim=3).unsqueeze(0)
            regular_grid.require_grads =False
            self.register_buffer('regular_grid',regular_grid.to(device =x.device, dtype=x.dtype))
            # shape = torch.tensor(shape).view(1,1,1,1,-1).clone().to(device =x.device, dtype=x.dtype)
            # shape.require_grads = False
            # self.register_buffer('shape',shape)
            self.first = False
        # print(self.regular_grid.shape)
        # print(self.regular_grid.dtype)
        # offset = torch.add(self.conv_deform(x).permute([0, 2,3,4, 1]) , self.regular_grid)
        # print(self.shape)
        shape = x.shape[2:]
        shape = torch.tensor(shape).view(1, 1, 1, 1, -1).to(device=x.device, dtype=x.dtype)

        offset = self.regular_grid+F.tanh(self.conv_deform(x).permute([0,2,3,4,1]))/shape
        #offset = torch.addcdiv(self.regular_grid, value=1, tensor1=self.conv_deform(x).permute([0,2,3,4,1]), tensor2=self.shape)
        x_offset = F.grid_sample(x, offset)
        out = self.conv(x_offset)
        return out

class Deform_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Deform_conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.conv_deform = nn.Conv2d(in_channels, 2, kernel_size=5, padding=2)
        self.conv_deform.weight.data.fill_(0)
        self.conv_deform.bias.data.fill_(0)
        self.regular_grid = None


    def forward(self, x):
        if self.regular_grid is None:
            shape = x.shape[2:]
            xx, yy = torch.meshgrid([torch.linspace(-1, 1, shape[0]), torch.linspace(-1, 1, shape[1])])
            self.regular_grid = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], dim=2).unsqueeze(0)
        offset = self.conv_deform(x).permute([0, 2,3, 1])+self.regular_grid
        x_offset = F.grid_sample(x, offset)
        out = self.conv(x_offset)
        return out

if __name__ == '__main__':
    input = torch.zeros(2, 10, 20,20,20)
    d_conv = Deform_conv3d(10, 20, kernel_size=3, padding=1)
    out = d_conv(input)
    print(out.shape)
