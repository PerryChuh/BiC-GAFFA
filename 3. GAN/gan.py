#%% 
import os, time, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from mixture_gaussian import dis_draw, GMM, plot_samples

from models import Generator0 as Generator
from models import Discriminator0 as Discriminator
from utils import OT_err as OT_matrice
    
#%%
class opts():
    def __init__(self):
        self.name = "GAN"
        self.n_steps = 2701
        self.batch_size = 512
        self.latent_dim = 128
        self.gpu = 0
        self.d_lr = 1e-4
        self.g_lr = 1e-3
        self.b1 = 0.5
        self.b2 = 0.999
        self.z_dim = 256
        self.d_steps = 5
        self.g_steps = 1
        self.seed = 0
        self.nGaussians = 8
        self.prefix = "images/GAN_Vanila" + f"_dlr={self.d_lr}_glr={self.g_lr}_dloop={self.d_steps}_seed={self.seed}_nGaussians={self.nGaussians}"

    def update_prefix(self):
        self.prefix = "images/GAN_Vanila" + f"_dlr={self.d_lr}_glr={self.g_lr}_dloop={self.d_steps}_seed={self.seed}_nGaussians={self.nGaussians}"

def main(opt=None):
    os.makedirs("images", exist_ok=True)
    plt.style.use('ggplot')

    opt = opts() if opt is None else opt
    opt.prefix = f"images/GAN_Vanila" + f"_dlr={opt.d_lr}_glr={opt.g_lr}_dloop={opt.d_steps}_seed={opt.seed}_nGaussians={opt.nGaussians}"

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:"+f"{opt.gpu}" if cuda else "cpu")

    # data generator and metric
    rng = np.random.default_rng(opt.seed)
    P = rng.random(opt.nGaussians)
    P /= np.sum(P)
    gmm = GMM(P)

    OT_err = lambda generator: OT_matrice(generator, gmm=gmm, z_dim=opt.z_dim, device=device, N=1000)

    # Initialize generator and discriminator
    generator = Generator(input_size=256, hidden_size=128, output_size=2)
    discriminator = Discriminator(input_size=2, hidden_size=128, output_size=1)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    if cuda:
        generator.to(device)
        discriminator.to(device)
        adversarial_loss.to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))

    def d_loop():
        valid = torch.ones((opt.batch_size, 1)).to(device)
        fake = torch.zeros((opt.batch_size, 1)).to(device)
        optimizer_D.zero_grad()

        # Configure input
        real_imgs = gmm.sample((opt.batch_size,)).to(device)

        # Sample noise as generator input
        gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)

        # Generate a batch of images
        with torch.no_grad():
            gen_imgs = generator(gen_input)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
        d_loss = (real_loss + fake_loss)/ 2

        d_loss.backward()
        optimizer_D.step()
        return real_loss.cpu().item(), fake_loss.cpu().item()

    def g_loop():
        valid = torch.ones((opt.batch_size, 1)).to(device)
        optimizer_G.zero_grad()

        # Sample noise as generator input
        gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)

        # Generate a batch of images
        gen_imgs = generator(gen_input)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()
        return g_loss.cpu().item()   

    EMs = []
    Time = []
    losses_d = []
    losses_g = []
    T = 0

    for step in range(opt.n_steps):
        t0 = time.time()

        d_infos = []
        for d_index in range(opt.d_steps):
            d_info = d_loop()
            d_infos.append(d_info)
        d_infos = np.mean(d_infos, 0)
        d_real_loss, d_fake_loss = d_infos

        g_infos = []
        for g_index in range(opt.g_steps):
            g_info = g_loop()
            g_infos.append(g_info)
        g_infos = np.mean(g_infos)
        g_loss = g_infos

        T += time.time() - t0
        Time.append(T)
        losses_d.append((d_real_loss + d_fake_loss).item()/2)
        losses_g.append(g_loss.item())

        EM_distance = OT_err(generator)
        EMs.append(EM_distance)

        sys.stdout.write(
            "%10s [step %5d/%5d] [D loss: %.2f] [G loss: %.2f] [OT loss: %.2f] \r"
            % ("GAN", step, opt.n_steps, losses_d[-1], losses_g[-1], EM_distance)
        )
        sys.stdout.flush()


    # save data
    print()
    pd.DataFrame({"Time": Time, "EM": EMs, "D_loss": losses_d, "G_loss": losses_g}).to_csv(opt.prefix + "_EM.csv")

#%%
if __name__ == "__main__":
    main()
