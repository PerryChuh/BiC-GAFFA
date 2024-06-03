#%% 
import os, time, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from mixture_gaussian import GMM, plot_samples

from models import Generator0 as Generator
from models import DiscriminatorR as Discriminator
from utils import OT_err as OT_matrice

import copy
#%%
class opts():
    def __init__(self):
        self.name = "UGAN"
        self.n_steps = 2701
        self.batch_size = 512
        self.latent_dim = 128
        self.z_dim = 256
        self.gpu = 0
        self.d_lr = 1e-4
        self.g_lr = 1e-3
        self.b1 = 0.5
        self.b2 = 0.999
        self.d_steps = 1
        self.g_steps = 1
        self.unrolled_steps = 5
        self.seed = 0
        self.nGaussians = 8
        self.prefix = "images/UGAN" + f"_unrolled_steps={self.unrolled_steps}_dlr={self.d_lr}_glr={self.g_lr}_dloop={self.d_steps}_seed={self.seed}_nGaussians={self.nGaussians}"

    def update_prefix(self):
        self.prefix = "images/UGAN" + f"_unrolled_steps={self.unrolled_steps}_dlr={self.d_lr}_glr={self.g_lr}_dloop={self.d_steps}_seed={self.seed}_nGaussians={self.nGaussians}"

def main(opt=None):
    os.makedirs("images", exist_ok=True)
    plt.style.use('ggplot')

    opt = opts() if opt is None else opt
    opt.prefix = "images/UGAN" + f"_unrolled_steps={opt.unrolled_steps}_dlr={opt.d_lr}_glr={opt.g_lr}_dloop={opt.d_steps}_seed={opt.seed}_nGaussians={opt.nGaussians}"

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:"+f"{opt.gpu}" if cuda else "cpu")

    # data generator and metric
    rng = np.random.default_rng(opt.seed)
    torch.manual_seed = opt.seed
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
        # 1, Train D on real+fake
        optimizer_D.zero_grad()

        #  1A: Train D on real
        d_real_data = gmm.sample((opt.batch_size,)).to(device)
        d_real_decision = discriminator(d_real_data)
        target = torch.ones((opt.batch_size, 1)).to(device)
        d_real_error = adversarial_loss(d_real_decision, target)  # ones = true

        #  1B: Train D on fake
        d_gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)
            
        with torch.no_grad():
            d_fake_data = generator(d_gen_input)
        d_fake_decision = discriminator(d_fake_data)
        target = torch.zeros((opt.batch_size, 1)).to(device)
        d_fake_error = adversarial_loss(d_fake_decision, target)  # zeros = fake
        
        d_loss = (d_real_error + d_fake_error)/2
        d_loss.backward()
        optimizer_D.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
        return d_real_error.cpu().item(), d_fake_error.cpu().item()

    def d_unrolled_loop(d_gen_input=None, second_order=False):
        # 1, Train D on real+fake
        optimizer_D.zero_grad()

        #  1A: Train D on real
        # d_real_data = gmm.sample((opt.batch_size,)).type(Tensor)
        d_real_data = gmm.sample((opt.batch_size,)).to(device)
        d_real_decision = discriminator(d_real_data)
        # target = Tensor(opt.batch_size, 1).fill_(1.0)
        target = torch.ones((opt.batch_size, 1)).to(device)
        d_real_error = adversarial_loss(d_real_decision, target)  # ones = true

        #  1B: Train D on fake
        if d_gen_input is None:
            # d_gen_input = Tensor(np.random.normal(0, 1, (opt.batch_size, opt.z_dim)))
            d_gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)
            
        with torch.no_grad():
            d_fake_data = generator(d_gen_input)
        d_fake_decision = discriminator(d_fake_data)
        # target = Tensor(opt.batch_size, 1).fill_(0.0)
        target = torch.zeros((opt.batch_size, 1)).to(device)
        d_fake_error = adversarial_loss(d_fake_decision, target)  # zeros = fake
        
        d_loss = (d_real_error + d_fake_error)/2
        d_loss.backward(create_graph = second_order)
        optimizer_D.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
        return None

    def g_loop(unrolled_steps = 0, second_order=False):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        optimizer_G.zero_grad()

        # gen_input = Tensor(np.random.normal(0, 1, (opt.batch_size, opt.z_dim)))
        gen_input = torch.from_numpy(np.random.normal(size = (opt.batch_size, opt.z_dim)).astype("float32")).to(device)
            
        if unrolled_steps > 0:
            backup = copy.deepcopy(discriminator)
            for i in range(unrolled_steps):
                d_unrolled_loop(d_gen_input=gen_input, second_order=second_order)
        
        g_fake_data = generator(gen_input)
        dg_fake_decision = discriminator(g_fake_data)
        # target = Tensor(opt.batch_size, 1).fill_(1.0)
        target = torch.ones((opt.batch_size, 1)).to(device)
        g_error = adversarial_loss(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
        # target = Tensor(opt.batch_size, 1).fill_(0.0)
        # g_error = -adversarial_loss(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
        g_error.backward()
        optimizer_G.step()  # Only optimizes G's parameters
        
        if unrolled_steps > 0:
            discriminator.load(backup)    
            del backup
        return g_error.cpu().item()

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
            g_info = g_loop(opt.unrolled_steps)
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
            % ("Unroll GAN", step, opt.n_steps, losses_d[-1], losses_g[-1], EM_distance)
        )
        sys.stdout.flush()


    print()
    # save data
    pd.DataFrame({"Time": Time, "EM": EMs, "D_loss": losses_d, "G_loss": losses_g}).to_csv(opt.prefix + "_EM.csv")

#%%
if __name__ == "__main__":
    main()