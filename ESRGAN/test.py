import torch
from torch import nn
from generator import Generator
from discriminator import Discriminator

def initialize_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

def test():
    gen = Generator()
    disc = Discriminator()
    low_res = 24
    x = torch.randn((5, 3, low_res, low_res))
    gen_out = gen(x)
    disc_out = disc(gen_out)

    print(f"Generator Output Shape: \n {gen_out.shape}")    
    print(f"Discriminator Output Shape: \n {disc_out.shape}")    

if __name__ == "__main__":
    test()