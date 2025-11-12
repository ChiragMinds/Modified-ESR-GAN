class Generator(nn.Module):
    def __init__(self, in_channels=1, num_channels=32, num_blocks=10):  # in_channels=1 for grayscale
        super().__init__()
        self.initial = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.residuals = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=0.5, mode='nearest')
        )
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        initial = self.initial(x)
        x = self.conv(self.residuals(initial)) + initial
        x = self.upsample(x)
        return self.final(x)

class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(2)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x

class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=16, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()
        for i in range(3):
            self.blocks.append(ConvBlock(in_channels + channels * i, channels if i <= 1 else in_channels, 3, 1, 1, use_act=True if i <= 1 else False))

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.residual_beta * out + x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_act=True):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_channels=32):  # in_channels=1 for grayscale
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
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1).mean(1).unsqueeze(1)

# Pre-trained VGG19 for perceptual loss
class VGG19PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[0, 5, 10, 19, 28]):
        super(VGG19PerceptualLoss, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.layers = [vgg19[i] for i in feature_layers]
        self.model = nn.Sequential(*self.layers).to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Ensure input images have 3 channels
        if x.size(1) != 3:
            x = x.repeat(1, 3, 1, 1)
        if y.size(1) != 3:
            y = y.repeat(1, 3, 1, 1)

        # Extract features
        x_vgg = self.model(x)
        y_vgg = self.model(y)

        # Compute perceptual loss
        return mse_loss(x_vgg, y_vgg)
