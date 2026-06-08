# model.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)                      # agrandit x ×2
        x = torch.cat([skip, x], dim=1)     # concatène skip + x sur les canaux
        return self.conv(x)                 # réduit les canaux


class UNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        # Encodeur
        self.inc    = DoubleConv(3, 64)
        self.down1  = Down(64, 128)
        self.down2  = Down(128, 256)
        self.down3  = Down(256, 512)

        # Goulot
        self.bottleneck = Down(512, 1024)

        # Décodeur
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Sortie
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)          # [3,128,128]  → [64,128,128]
        x2 = self.down1(x1)       # [64,128,128] → [128,64,64]
        x3 = self.down2(x2)       # [128,64,64]  → [256,32,32]
        x4 = self.down3(x3)       # [256,32,32]  → [512,16,16]
        x5 = self.bottleneck(x4)  # [512,16,16]  → [1024,8,8]

        x = self.up1(x5, x4)      # [1024,8,8]   → [512,16,16]
        x = self.up2(x,  x3)      # [512,16,16]  → [256,32,32]
        x = self.up3(x,  x2)      # [256,32,32]  → [128,64,64]
        x = self.up4(x,  x1)      # [128,64,64]  → [64,128,128]

        return self.outc(x)        # [64,128,128] → [3,128,128]