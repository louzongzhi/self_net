import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(self_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        channels = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

        self.inc = (DoubleConv(n_channels, channels[0]))

        self.down1 = (Down(channels[0], channels[1]))
        self.down2 = (Down(channels[1], channels[2]))
        self.down3 = (Down(channels[2], channels[3]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(channels[3], channels[4] // factor))
        self.down5 = (Down(channels[4], channels[5] // factor))
        self.down6 = (Down(channels[5], channels[6] // factor))
        self.down7 = (Down(channels[6], channels[7] // factor))
        self.down8 = (Down(channels[7], channels[8] // factor))
        self.down9 = (Down(channels[8], channels[9] // factor))
        self.down10 = (Down(channels[9], channels[10] // factor))
        self.down11 = (Down(channels[10], channels[11] // factor))

        self.up1 = (Up(channels[11], channels[10] // factor, bilinear))
        self.up2 = (Up(channels[10], channels[9] // factor, bilinear))
        self.up3 = (Up(channels[9], channels[8] // factor, bilinear))
        self.up4 = (Up(channels[8], channels[7] // factor, bilinear))
        self.up5 = (Up(channels[7], channels[6] // factor, bilinear))
        self.up6 = (Up(channels[6], channels[5] // factor, bilinear))
        self.up7 = (Up(channels[5], channels[4] // factor, bilinear))
        self.up8 = (Up(channels[4], channels[3] // factor, bilinear))
        self.up9 = (Up(channels[3], channels[2] // factor, bilinear))
        self.up10 = (Up(channels[2], channels[1] // factor, bilinear))
        self.up11 = (Up(channels[1], channels[0] // factor, bilinear))

        self.outc = OutConv(channels[0] // factor, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        x0 = x
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        x9 = self.down9(x8)
        x10 = self.down10(x9)
        x11 = self.down11(x10)

        x = self.up1(x11, x10)
        x = self.up2(x, x9)
        x = self.up3(x, x8)
        x = self.up4(x, x7)
        x = self.up5(x, x6)
        x = self.up6(x, x5)
        x = self.up7(x, x4)
        x = self.up8(x, x3)
        x = self.up9(x, x2)
        x = self.up10(x, x1)
        x = self.up11(x, x0)

        logits = self.outc(x)
        return logits
