import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels, out_channels,
            **kwargs,
            bias=True
        )
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor, mode="nearest")
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            3, 1, 1, bias=True
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))
    
class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels = 32, residual_beta = 0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels * i,
                    channels if i <= 3 else in_channels,
                    kernel_size = 3,
                    stride = 1, padding = 1,
                    use_act=True if i <=3 else False
                )
            )

    def forward(self, x):
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_input = torch.cat([new_inputs, out], dim=1)
        
        return self.residual_beta * out + x
    
class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta = 0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(
            *[DenseResidualBlock(in_channels) for _ in range(3)]
        )

    def forward(self, x):
        return self.rrdb(x) + self.residual_beta + x
    

class Generator(nn.Module):
    def __init__(self, in_channels = 3, num_channels = 64, num_blocks = 23):
        super().__init__()
        self.inital = nn.Conv2d(
            in_channels,
            num_channels,
            3, 1, 1,
            bias=True
        )
        self.residuals = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(num_channels , num_channels, 3, 1, 1)
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels), UpsampleBlock(num_channels)
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        initial = self.inital(x)
        x = self.conv(self.residuals(initial)) + initial
        x = self.upsamples(x)
        return self.final(x)
