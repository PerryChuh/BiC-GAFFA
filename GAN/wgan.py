#%% 
import os, time, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from mixture_gaussian import dis_draw, GMM, plot_samples

from models import GeneratorW as Generator
from models import DiscriminatorW as Discriminator
from utils import OT_err as OT_matrice
    
#%%
class opts():
    def __init__(self):
        self.name = "WGAN"
        self.n_steps = 2701
        self.batch_size = 512
        self.latent_dim = 128
        self.z_dim = 256
        self.gpu = 0
        self.d_lr = 1e-4
        self.g_lr = 1e-3
        self.b1 = 0.5
        self.b2 = 0.999
        self.clip_value = 0.001
        self.d_steps = 5
        self.seed = 0
        self.nGaussians = 8
        self.prefix = "images/WGAN" + f"_dlr={self.d_lr}_glr={self.g_lr}_dloop={self.d_steps}_clip={self.clip_value}_seed={self.seed}_nGaussians={self.nGaussians}"
        
    def update_prefix(self):
        self.prefix = "images/WGAN" + f"_dlr={self.d_lr}_glr={self.g_lr}_dloop={self.d_steps}_clip={self.clip_value}_seed={self.seed}_nGaussians={self.nGaussians}"

def main(opt=None):
    os.makedirs("images", exist_ok=True)
    plt.style.use('ggplot')

    opt = opts() if opt is None else opt
    opt.prefix = "images/WGAN" + f"_dlr={opt.d_lr}_glr={opt.g_lr}_dloop={opt.d_steps}_clip={opt.clip_value}_seed={opt.seed}_nGaussians={opt.nGaussians}"

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:"+f"{opt.gpu}" if cuda else "cpu")

    # Initialize generator and discriminator
    generator = Generator(input_size=256, hidden_size=128, output_size=2)
    discriminator = Discriminator(input_size=2, hidden_size=128, output_size=1)

    if cuda:
        generator.to(device)
        discriminator.to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
    # optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.g_lr)
    # optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.d_lr)

    rng = np.random.default_rng(opt.seed)
    torch.manual_seed = opt.seed
    P = rng.random(opt.nGaussians)
    P /= np.sum(P)
    gmm = GMM(P)

    OT_err = lambda generator: OT_matrice(generator, gmm=gmm, z_dim=opt.z_dim, device=device, N=1000)

    EMs = []
    Time = []
    T = 0

    for step in range(opt.n_steps):
        t0 = time.time()

        # Adversarial ground truths
        # ---------------------
        #  Train Discriminator
        # ---------------------

        for _ in range(opt.d_steps):
            optimizer_D.zero_grad()

            # Configure input
            # imgs = torch.from_numpy(gmm.sample((opt.batch_size,)))
            imgs = gmm.sample((opt.batch_size,))
            # real_imgs = imgs.type(Tensor)
            real_imgs = imgs.to(device)

            # Sample noise as generator input
            gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)

            # Generate a batch of images
            with torch.no_grad():
                gen_imgs = generator(gen_input)

            # Measure discriminator's ability to classify real from generated samples
            d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(gen_imgs))

            d_loss.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)

        # Generate a batch of images
        gen_imgs = generator(gen_input)

        # Loss measures generator's ability to fool the discriminator
        g_loss = -torch.mean(discriminator(gen_imgs))

        g_loss.backward()
        optimizer_G.step()

        T += time.time() - t0
        Time.append(T)

        EM_distance = OT_err(generator)
        EMs.append(EM_distance)

        sys.stdout.write(
            "%10s [step %5d/%5d] [D loss: %.2f] [G loss: %.2f] [OT loss: %.2f] \r"
            % ("WGAN", step, opt.n_steps, d_loss.item() / 2, g_loss.item(), EM_distance)
        )
        sys.stdout.flush()


    print()
    pd.DataFrame({"Time": Time, "EM": EMs}).to_csv(opt.prefix + "_EM.csv")


if __name__ == "__main__":
    main()