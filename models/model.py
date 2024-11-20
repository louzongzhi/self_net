import torch
import torch.nn as nn
import torch.nn.functional as F
from parts.conv import ACConv2d


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ACConv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            ACConv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class self_net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, deep_supervision=False):
        super(self_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision
        channels = [32, 64, 128, 256, 512]

        self.inc = (DoubleConv(n_channels, channels[0]))

        self.down1 = (Down(channels[0], channels[1]))
        self.down2 = (Down(channels[1], channels[2]))
        self.down3 = (Down(channels[2], channels[3]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(channels[3], channels[4] // factor))

        self.up1 = (Up(channels[4], channels[3] // factor, bilinear))
        self.up2 = (Up(channels[3], channels[2] // factor, bilinear))
        self.up3 = (Up(channels[2], channels[1] // factor, bilinear))
        self.up4 = (Up(channels[1], channels[0] // factor, bilinear))

        self.outc = OutConv(channels[0] // factor, n_classes)

    def forward(self, x):
        x_0 = self.inc(x)
        x_1 = self.down1(x_0)
        x_2 = self.down2(x_1)
        x_3 = self.down3(x_2)
        x_4 = self.down4(x_3)

        x = self.up1(x_4, x_3)
        x = self.up2(x, x_2)
        x = self.up3(x, x_1)
        x = self.up4(x, x_0)

        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    model = self_net(3, 4, bilinear=False)
    input = torch.rand(1, 3, 200, 200)
    output = model(input)
    print(output.shape)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}\tM')