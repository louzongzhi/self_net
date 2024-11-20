import torch
import torch.nn as nn
import math


class ACConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, branch_ratio=0.125):
        super(ACConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias, groups=math.gcd(in_channels, out_channels))
        self.ac1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                             stride=stride, padding=(0, padding), bias=bias)
        self.ac2 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                             stride=stride, padding=(padding, 0), bias=bias)

    def forward(self, x):
        ac1 = self.ac1(x)
        ac2 = self.ac2(x)
        x = self.conv(x)
        return (ac1 + ac2 + x) / 3


if __name__ == '__main__':
    model = ACConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
    input = torch.randn(32, 3, 200, 200)
    output = model(input)
    print(f'{output.shape}\n{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}\tM')