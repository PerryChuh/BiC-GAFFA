#%% 
import os, time, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from mixture_gaussian import dis_draw, GMM, plot_samples

from models import Generator0 as Generator
from models import Discriminator0 as Discriminator
from utils import OT_err as OT_matrice

#%%
class opts():
    def __init__(self):
        self.name = "Bi-WGAN"
        self.n_steps = 2701
        self.batch_size = 512
        self.latent_dim = 128
        self.z_dim = 256
        self.gpu = 0
        self.d2_lr = 1e-3
        self.d1_lr = 1e-4
        self.z_lr = 1e-4
        self.g_lr = 1e-3
        self.b1 = 0.5
        self.b2 = 0.999
        self.gam1 = 10.
        self.gam2 = 1. 
        self.d2_steps = 5
        self.method = 2 
        self.p = 0.1
        self.c0 = 1
        self.seed = 0
        self.nGaussians = 8
        self.prefix = f"images/BiwGAN_method={self.method}_d2steps={self.d2_steps}_gam1={self.gam1}_gam2={self.gam2}_c0={self.c0}_p={self.p}_d1lr={self.d1_lr}_d2lr={self.d2_lr}_glr={self.g_lr}_zlr={self.z_lr}_seed={self.seed}_nGaussians={self.nGaussians}"

    def update_prefix(self):
        self.prefix = f"images/BiwGAN_method={self.method}_d2steps={self.d2_steps}_gam1={self.gam1}_gam2={self.gam2}_c0={self.c0}_p={self.p}_d1lr={self.d1_lr}_d2lr={self.d2_lr}_glr={self.g_lr}_zlr={self.z_lr}_seed={self.seed}_nGaussians={self.nGaussians}"

def main(opt=None):
    os.makedirs("images", exist_ok=True)
    plt.style.use('ggplot')

    opt = opts() if opt is None else opt
    opt.prefix = f"images/BiwGAN_method={opt.method}_d2steps={opt.d2_steps}_gam1={opt.gam1}_gam2={opt.gam2}_c0={opt.c0}_p={opt.p}_d1lr={opt.d1_lr}_d2lr={opt.d2_lr}_glr={opt.g_lr}_zlr={opt.z_lr}_seed={opt.seed}_nGaussians={opt.nGaussians}"

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:"+f"{opt.gpu}" if cuda else "cpu")

    # Initialize generator and discriminator
    generator = Generator(input_size=256, hidden_size=128, output_size=2)
    discriminator1 = Discriminator(input_size=2, hidden_size=128, output_size=1)
    discriminator2 = Discriminator(input_size=2, hidden_size=128, output_size=1)

    if cuda:
        generator.to(device)
        discriminator1.to(device)
        discriminator2.to(device)

    lamk = torch.tensor([0.1]).to(device)
    zk = torch.tensor([0.1]).to(device).requires_grad_()

    # Optimizers
    optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=opt.d2_lr, betas=(opt.b1, opt.b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=opt.d1_lr, betas=(opt.b1, opt.b2))
    optimizer_z = torch.optim.Adam([zk], lr=opt.z_lr, betas=(opt.b1, opt.b2))

    rng = np.random.default_rng(opt.seed)
    torch.manual_seed = opt.seed
    P = rng.random(opt.nGaussians)
    P /= np.sum(P)
    gmm = GMM(P)
    OT_err = lambda generator, N: OT_matrice(generator, gmm=gmm, z_dim=opt.z_dim, device=device, N=N)

    def diff_loss(D1, D2):
        return sum(
            [torch.sum((p1 - p2).pow(2)) for p1, p2 in zip(D1.parameters(), D2.parameters())]
        )

    d_criterion = lambda D, real, fake: -torch.mean(torch.log(D(real)) + torch.log(1 - D(fake)))
    g_criterion = lambda D, fake: -torch.mean(torch.log(D(fake)))

    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1)).astype("float32")).to(device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones((real_samples.shape[0], 1)).to(device)

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
        # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty = (gradients.norm(2, dim=1)).max() - 1.0
        return gradient_penalty

    def g2_loop():
        optimizer_D2.zero_grad()


        # Configure input
        real_imgs = gmm.sample((opt.batch_size, )).to(device)

        # Sample noise as generator input
        # gen_input = Tensor(np.random.normal(size = (opt.batch_size, opt.z_dim)))
        gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)

        # Generate a batch of images
        with torch.no_grad():
            gen_imgs = generator(gen_input)

        # Measure discriminator's ability to classify real from generated samples
        g = compute_gradient_penalty(discriminator2, real_imgs.data, gen_imgs.data)
        # d2_loss = real_loss2 + fake_loss2
        d2_loss = d_criterion(discriminator2, real_imgs, gen_imgs)

        reg = 1/opt.gam1 * diff_loss(discriminator1, discriminator2)

        (d2_loss + zk * g + reg).backward()
        optimizer_D2.step()

        return d2_loss.cpu().item()

    def dg_loop1(ck):
        optimizer_G.zero_grad()
        optimizer_D1.zero_grad()


        # Sample real data and sample noise for generator input 
        real_imgs = gmm.sample((opt.batch_size,)).to(device)
        # gen_imgs = generator(Tensor(np.random.normal(size = (opt.batch_size, opt.z_dim))))
        gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)
        gen_imgs = generator(gen_input)

        d1_loss = d_criterion(discriminator1, real_imgs, gen_imgs)
        d2_loss = d_criterion(discriminator2, real_imgs, gen_imgs)

        (d1_loss - d2_loss).backward(retain_graph=True)

        reg = - 1/opt.gam1 * diff_loss(discriminator1, discriminator2)
        reg.backward(retain_graph=True)

        g1 = compute_gradient_penalty(discriminator1, real_imgs, gen_imgs)
        g2 = compute_gradient_penalty(discriminator2, real_imgs, gen_imgs)

        (lamk * g1).backward(retain_graph=True)
        (-zk * g2).backward(retain_graph=True)

        g_error1 = - d_criterion(discriminator1, real_imgs, gen_imgs) / ck
        g_error2 = - d_criterion(discriminator1, real_imgs, gen_imgs) / ck
        (g_error1 + g_error2).backward()
        g_error = - g_error1

        # g_error = adversarial_loss(discriminator1(gen_imgs), valid) / ck
        # g_error.backward()

        optimizer_D1.step()
        optimizer_G.step()

        return d1_loss.cpu().item(), g_error.cpu().item() * ck

    def dg_loop2(ck):
        optimizer_G.zero_grad()
        optimizer_D1.zero_grad()

        # Sample real data and sample noise for generator input 
        real_imgs = gmm.sample((opt.batch_size,)).to(device)
        # gen_imgs = generator(Tensor(np.random.normal(size = (opt.batch_size, opt.z_dim))))
        gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)
        gen_imgs = generator(gen_input)

        d1_loss = d_criterion(discriminator1, real_imgs, gen_imgs)
        d2_loss = d_criterion(discriminator2, real_imgs, gen_imgs)

        (d1_loss - d2_loss).backward(retain_graph=True)

        reg = - 1/opt.gam1 * diff_loss(discriminator1, discriminator2)
        reg.backward(retain_graph=True)

        g1 = compute_gradient_penalty(discriminator1, real_imgs, gen_imgs)
        g2 = compute_gradient_penalty(discriminator2, real_imgs, gen_imgs)

        (lamk * g1).backward(retain_graph=True)
        (-zk * g2).backward(retain_graph=True)

        g_error = g_criterion(discriminator1, gen_imgs) / ck
        g_error.backward()

        optimizer_D1.step()
        optimizer_G.step()

        return d1_loss.cpu().item(), g_error.cpu().item() * ck

    EMs = []
    Time = []
    T = 0

    for step in range(opt.n_steps):
        t0 = time.time()
        ck = min(opt.c0 * (step + 1) ** opt.p, 10)

        real_imgs = gmm.sample((opt.batch_size,)).to(device)
        gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)
        with torch.no_grad():
            gen_imgs = generator(gen_input)
        gk = compute_gradient_penalty(discriminator1, real_imgs, gen_imgs)
        lamk = torch.relu(zk + opt.gam2 * gk)

        d_losses = []
        for _ in range(opt.d2_steps):
            d_losses.append(g2_loop())
        d2_loss = np.mean(d_losses)
        
        if opt.method == 1:
            d1_loss, g_loss = dg_loop1(ck)
        elif opt.method == 2:
            d1_loss, g_loss = dg_loop2(ck)

        optimizer_z.zero_grad()
        with torch.no_grad():
            real_imgs = gmm.sample((opt.batch_size,)).to(device)
            gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)
            gen_imgs = generator(gen_input)
        gk = compute_gradient_penalty(discriminator2, real_imgs, gen_imgs)
        z_grad = - (zk - lamk) / opt.gam2 - zk * gk
        zk.grad = z_grad
        optimizer_z.step()
        zk.data.clamp_(0., .1)

        T += time.time() - t0
        Time.append(T)

        EM_distance = OT_err(generator, N = 1000)
        EMs.append(EM_distance)

        sys.stdout.write(
            "%10s [step %5d/%5d] [D1 loss: %.2f] [D2 loss: %.2f] [G loss: %.2f] [gk: %4.2f] [OT loss: %.2f] %.2f\r"
            % ("Bi-WGAN", step, opt.n_steps, d1_loss, d2_loss, g_loss, gk, EM_distance, zk.item())
        )
        sys.stdout.flush()


    print()
    pd.DataFrame({"Time": Time, "EM": EMs}).to_csv(opt.prefix + "_EM.csv", index=False)

#%%
if __name__ == "__main__":
    main()