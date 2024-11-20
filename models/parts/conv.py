import torch
import torch.nn as nn
import math


class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super(InceptionDWConv2d, self).__init__()
        self.branch_channels = max(1, int(in_channels * branch_ratio))
        self.square_kernel_size = square_kernel_size
        self.band_kernel_size = band_kernel_size
        self.dwconv_hw = nn.Conv2d(self.branch_channels, self.branch_channels,
                                   kernel_size=square_kernel_size, padding=square_kernel_size//2, groups=self.branch_channels)
        self.dwconv_w = nn.Conv2d(self.branch_channels, self.branch_channels,
                                  kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=self.branch_channels)
        self.dwconv_h = nn.Conv2d(self.branch_channels, self.branch_channels,
                                  kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=self.branch_channels)
        self.split_indexes = (in_channels - 3 * self.branch_channels, self.branch_channels, self.branch_channels, self.branch_channels)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)
        return torch.cat((x_id, x_hw, x_w, x_h), dim=1)


class ACConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, branch_ratio=0.125):
        super(ACConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias, groups=math.gcd(in_channels, out_channels))
        self.ac1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                             stride=stride, padding=(0, padding), bias=bias)
        self.ac2 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                             stride=stride, padding=(padding, 0), bias=bias)
        self.inception_dwconv = InceptionDWConv2d(in_channels, branch_ratio=branch_ratio)

    def forward(self, x):
        x = self.inception_dwconv(x)
        ac1 = self.ac1(x)
        ac2 = self.ac2(x)
        x = self.conv(x)
        return (ac1 + ac2 + x) / 3


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
#         size_tensor = x.size()
#         return torch.cat([x[...,0:size_tensor[2]//2,0:size_tensor[3]//2],
#                          x[...,0:size_tensor[2]//2,size_tensor[3]//2:],
#                          x[...,size_tensor[2]//2:,0:size_tensor[3]//2],
#                          x[...,size_tensor[2]//2:,size_tensor[3]//2:]  ],1)


if __name__ == '__main__':
    model = ACConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
    input = torch.randn(1, 3, 32, 32)
    output = model(input)
    output = space_to_depth()(output)
    print(f'{output.shape}\n{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}\tM')