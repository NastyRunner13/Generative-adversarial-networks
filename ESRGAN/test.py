import torch
from generator import Generator
from discriminator import Discriminator

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