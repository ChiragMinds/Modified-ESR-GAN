# src/esrgan/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

# ---------------------------
# Basic conv block
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_act=True):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))


# ---------------------------
# DenseResidualBlock used in RRDBs
# ---------------------------
class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=16, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()
        # three convs with concatenations like ESRGAN's dense block
        for i in range(3):
            out_ch = channels if i <= 1 else in_channels
            self.blocks.append(ConvBlock(in_channels + channels * i, out_ch, 3, 1, 1, use_act=(i <= 1)))

    def forward(self, x):
        new_inputs = x
        out = None
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        # residual connection
        return self.residual_beta * out + x


# ---------------------------
# RRDB (Residual in Residual Dense Block)
# ---------------------------
class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        # two DenseResidualBlock stacked (you can set 3 for original)
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(2)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x


# ---------------------------
# Upsample block (helper)
# ---------------------------
class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


# ---------------------------
# Generator
# ---------------------------
class Generator(nn.Module):
    def __init__(self, in_channels=1, num_channels=32, num_blocks=10):
        """
        in_channels: 1 for grayscale input
        num_channels: base channel width
        num_blocks: number of RRDB blocks
        """
        super().__init__()
        self.initial = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.residuals = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

        # Upsample sequence - note: your original used a stray scale_factor=0.5 at the end,
        # here we upsample twice (x2 then x2) to achieve x4 if needed; adjust as required.
        self.upsample = nn.Sequential(
            UpsampleBlock(num_channels, scale_factor=2),
            UpsampleBlock(num_channels, scale_factor=2),
        )
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        initial = self.initial(x)
        x = self.conv(self.residuals(initial)) + initial
        x = self.upsample(x)
        return self.final(x)


# ---------------------------
# Discriminator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_channels=32):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_channels, num_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_channels * 4, num_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_channels * 8, 1, kernel_size=4, stride=1, padding=0),
            # no activation here if using BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1).mean(1).unsqueeze(1)


# ---------------------------
# VGG19 perceptual model wrapper
# ---------------------------
class VGG19PerceptualLoss(nn.Module):
    def __init__(self, device=torch.device('cpu'), feature_layers=[0, 5, 10, 19, 28]):
        """
        feature_layers: indices of layers to extract from vgg19.features
        device: pass training device when instantiating to place model on same device
        """
        super().__init__()
        # support older/newer torchvision API
        try:
            vgg19 = tv_models.vgg19(pretrained=True).features
        except Exception:
            # newer torchvision uses weights parameter
            try:
                vgg19 = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
            except Exception:
                vgg19 = tv_models.vgg19(pretrained=True).features

        # pick layers
        self.layers = [vgg19[i] for i in feature_layers]
        self.model = nn.Sequential(*self.layers).to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Expect x,y in range [0,1] or normalized; ensure 3 channels for VGG
        if x.size(1) != 3:
            x = x.repeat(1, 3, 1, 1)
        if y.size(1) != 3:
            y = y.repeat(1, 3, 1, 1)

        x_vgg = self.model(x)
        y_vgg = self.model(y)
        return F.mse_loss(x_vgg, y_vgg)
