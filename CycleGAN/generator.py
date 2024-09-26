import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                padding_mode="reflect",
                **kwargs
            ) 
            if down
            else nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size = 3,
                padding = 1
            ),
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                use_act=False,
                kernel_size = 3,
                padding = 1
            )
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, img_channels,num_features=64, num_residuals = 9):
        super().__init__()
        self.inital = nn.Sequential(
            nn.Conv2d(
                in_channels = img_channels,
                out_channels = 64,
                kernel_size = 7,
                stride = 1,
                padding = 3,
                padding_mode = "reflect"  
            )
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=num_features,
                    out_channels=num_features*2,
                    kernel_size = 3,
                    stride = 2,
                    padding = 1
                ),
                ConvBlock(
                    in_channels=num_features*2,
                    out_channels=num_features*4,
                    kernel_size = 3,
                    stride = 2,
                    padding = 1
                )
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels=num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=num_features*4,
                    out_channels=num_features*2,
                    down=False,
                    kernel_size = 3,
                    stride = 2,
                    padding = 1,
                    output_padding = 1
                ),
                ConvBlock(
                    in_channels=num_features*2,
                    out_channels=num_features,
                    down=False,
                    kernel_size = 3,
                    stride = 2,
                    padding = 1,
                    output_padding = 1
                )
            ]
        )

        self.last = nn.Conv2d(
            in_channels = num_features,
            out_channels = img_channels,
            kernel_size=7,
            stride = 1,
            padding=3,
            padding_mode="reflect"
        )

    def forward(self, x):
        x = self.inital(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))
    
def test():
    img_channels = 3
    img_size  = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels=3, num_features=64, num_residuals=9)
    print(gen)
    print(gen(x).shape)

if __name__ == "__main__":
    test()