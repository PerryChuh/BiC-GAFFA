#%% 
import os, time, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from models import MNISTGenerator, MNISTDiscriminator
from utils import load_mnist
    
#%%
class opts():
    def __init__(self):
        self.name = "WGAN-MNIST"
        self.n_epochs = 100
        self.batch_size = 64
        self.latent_dim = 100
        self.img_shape = (1, 28, 28)
        self.gpu = 0
        self.d_lr = 0.0002
        self.g_lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.clip_value = 0.01
        self.d_steps = 5
        self.seed = 42
        self.sample_interval = 10
        self.data_path = './data'
        self.result_dir = "images"
        self.prefix = f"images/WGAN_MNIST_dlr={self.d_lr}_glr={self.g_lr}_dloop={self.d_steps}_clip={self.clip_value}_seed={self.seed}"
        
    def update_prefix(self):
        self.prefix = f"images/WGAN_MNIST_dlr={self.d_lr}_glr={self.g_lr}_dloop={self.d_steps}_clip={self.clip_value}_seed={self.seed}"

def main(opt=None):
    os.makedirs("images", exist_ok=True)
    plt.style.use('ggplot')

    opt = opts() if opt is None else opt
    opt.update_prefix()

    # 设置随机种子
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    
    # 配置设备
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:"+f"{opt.gpu}" if cuda else "cpu")
    
    # 加载MNIST数据集
    dataloader = load_mnist(batch_size=opt.batch_size, data_path=opt.data_path)
    
    # 初始化生成器和判别器
    generator = MNISTGenerator(latent_dim=opt.latent_dim, img_shape=opt.img_shape)
    discriminator = MNISTDiscriminator(img_shape=opt.img_shape)

    if cuda:
        generator.to(device)
        discriminator.to(device)

    # 优化器
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.g_lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.d_lr)

    # 训练记录
    losses = {"G": [], "D": []}
    training_time = []
    start_time = time.time()
    
    # 训练循环
    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for i, (imgs, _) in enumerate(dataloader):
            
            # 配置真实图像
            real_imgs = imgs.to(device)
            
            # ---------------------
            #  训练判别器
            # ---------------------
            for _ in range(opt.d_steps):
                optimizer_D.zero_grad()
                
                # 生成一批假图像
                z = torch.randn(imgs.shape[0], opt.latent_dim).to(device)
                fake_imgs = generator(z).detach()
                
                # 训练判别器
                # 判别器分数
                real_validity = discriminator(real_imgs)
                fake_validity = discriminator(fake_imgs)
                
                # 计算损失
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                
                # 反向传播和优化
                d_loss.backward()
                optimizer_D.step()
                
                # 裁剪判别器权重
                for p in discriminator.parameters():
                    p.data.clamp_(-opt.clip_value, opt.clip_value)
            
            # ---------------------
            #  训练生成器
            # ---------------------
            optimizer_G.zero_grad()
            
            # 生成一批图像
            z = torch.randn(imgs.shape[0], opt.latent_dim).to(device)
            gen_imgs = generator(z)
            
            # 判别器分数
            fake_validity = discriminator(gen_imgs)
            
            # 计算生成器损失
            g_loss = -torch.mean(fake_validity)
            
            # 反向传播和优化
            g_loss.backward()
            optimizer_G.step()
            
            # 记录损失
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            
            # 打印训练信息
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            sys.stdout.flush()
        
        # 计算平均损失
        epoch_d_loss /= len(dataloader)
        epoch_g_loss /= len(dataloader)
        losses["D"].append(epoch_d_loss)
        losses["G"].append(epoch_g_loss)
        
        # 记录训练时间
        epoch_time = time.time() - epoch_start_time
        training_time.append(time.time() - start_time)
        
        # 保存生成的图像样本
        if epoch % opt.sample_interval == 0:
            # 使用与bi_wgan.py相同的方式保存图像
            z = torch.randn(25, opt.latent_dim).to(device)
            with torch.no_grad():
                gen_imgs = generator(z)
            save_image(gen_imgs.data, f"{opt.result_dir}/wgan_epoch_{epoch}.png", nrow=5, normalize=True)
            print(f"\n已保存epoch {epoch}的生成图像到 {opt.result_dir}/wgan_epoch_{epoch}.png")
        
        print(f"\nEpoch {epoch}/{opt.n_epochs} completed in {epoch_time:.2f}s. D loss: {epoch_d_loss:.6f}, G loss: {epoch_g_loss:.6f}")
    
    # 保存模型
    torch.save(generator.state_dict(), f"{opt.prefix}_generator.pth")
    torch.save(discriminator.state_dict(), f"{opt.prefix}_discriminator.pth")
    
    # 保存训练记录
    pd.DataFrame({
        "Time": training_time,
        "D_Loss": losses["D"],
        "G_Loss": losses["G"]
    }).to_csv(f"{opt.result_dir}/wgan_training_log.csv", index=False)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses["D"], label="Discriminator")
    plt.plot(losses["G"], label="Generator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{opt.result_dir}/wgan_loss_curve.png")
    plt.close()

if __name__ == "__main__":
    main()