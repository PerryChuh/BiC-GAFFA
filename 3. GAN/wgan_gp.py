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
        self.name = "WGAN-GP"
        self.n_steps = 2701
        self.batch_size = 512
        self.latent_dim = 128
        self.z_dim = 256
        self.gpu = 0
        self.d_lr = 1e-4
        self.g_lr = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.lamgp = .1
        self.d_steps = 5
        self.seed = 0
        self.nGaussians = 8
        self.prefix = f"images/WGAN_GP_lamgp={self.lamgp}_dlr={self.d_lr}_glr={self.g_lr}_dloop={self.d_steps}_seed={self.seed}_nGaussians={self.nGaussians}"
        
    def update_prefix(self):
        self.prefix = f"images/WGAN_GP_lamgp={self.lamgp}_dlr={self.d_lr}_glr={self.g_lr}_dloop={self.d_steps}_seed={self.seed}_nGaussians={self.nGaussians}"

def main(opt=None):
    os.makedirs("images", exist_ok=True)
    plt.style.use('ggplot')

    opt = opts() if opt is None else opt
    opt.prefix = f"images/WGAN_GP_lamgp={opt.lamgp}_dlr={opt.d_lr}_glr={opt.g_lr}_dloop={opt.d_steps}_seed={opt.seed}_nGaussians={opt.nGaussians}"

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

    rng = np.random.default_rng(opt.seed)
    torch.manual_seed = opt.seed
    P = rng.random(opt.nGaussians)
    P /= np.sum(P)
    gmm = GMM(P)

    OT_err = lambda generator: OT_matrice(generator, gmm=gmm, z_dim=opt.z_dim, device=device, N=1000)

    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1)).astype("float32")).to(device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        # fake = torch.autograd.Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = torch.autograd.Variable(torch.ones((real_samples.shape[0], 1)).to(device), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    EMs = []
    Time = []
    T = 0

    for step in range(opt.n_steps):
        t0 = time.time()

        # Adversarial ground truths
        # valid = torch.ones((opt.batch_size, 1)).type(Tensor)
        # fake = torch.zeros((opt.batch_size, 1)).type(Tensor)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(opt.d_steps):
            optimizer_D.zero_grad()

            # Configure input
            imgs = gmm.sample((opt.batch_size,))
            real_imgs = imgs.to(device)

            # Sample noise as generator input
            gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)

            # Generate a batch of images
            with torch.no_grad():
                gen_imgs = generator(gen_input)

            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)

            # Measure discriminator's ability to classify real from generated samples
            d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(gen_imgs))

            (d_loss + opt.lamgp * gradient_penalty).backward()
            optimizer_D.step()

        # if step % opt.critc_iters == 0:
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
            "%10s [step %5d/%5d] [D loss: %.2f] [G loss: %.2f] [GP: %.2f] [OT loss: %.2f] \r"
            % ("WGAN-GP", step, opt.n_steps, d_loss.item(), g_loss.item(), gradient_penalty.item(), EM_distance)
        )
        sys.stdout.flush()


    print()
    pd.DataFrame({"Time": Time, "EM": EMs}).to_csv(opt.prefix + "_EM.csv")

if __name__ == "__main__":
    main()
# %%
