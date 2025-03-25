import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="训练的总轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
    parser.add_argument("--d1_lr", type=float, default=0.0002, help="判别器1学习率")
    parser.add_argument("--d2_lr", type=float, default=0.0001, help="判别器2(θ)学习率")
    parser.add_argument("--g_lr", type=float, default=0.0002, help="生成器学习率")
    parser.add_argument("--b1", type=float, default=0.9, help="Adam优化器beta1参数")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam优化器beta2参数")
    parser.add_argument("--latent_dim", type=int, default=100, help="噪声向量维度")
    parser.add_argument("--img_size", type=int, default=28, help="图像大小")
    parser.add_argument("--channels", type=int, default=1, help="通道数")
    parser.add_argument("--sample_interval", type=int, default=100, help="每n批次保存一次生成的样本")
    parser.add_argument("--save_freq", type=int, default=5, help="每n个epoch保存一次模型")
    parser.add_argument("--print_freq", type=int, default=50, help="每n批次打印一次训练信息")
    parser.add_argument("--gradient_penalty", type=float, default=10, help="梯度惩罚系数")
    parser.add_argument("--gam1", type=float, default=0.5, help="判别器1和判别器2(θ)之间的正则化系数")
    parser.add_argument("--gam2", type=float, default=0.5, help="拉格朗日乘子正则化系数")
    parser.add_argument("--rho1", type=float, default=1.0, help="惩罚参数1的初始值")
    parser.add_argument("--rho2", type=float, default=1.0, help="惩罚参数2的初始值")
    parser.add_argument("--penalty_increase_factor", type=float, default=1.1, help="惩罚参数增长因子")
    parser.add_argument("--penalty_update_freq", type=int, default=50, help="惩罚参数更新频率")
    parser.add_argument("--clip_value", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--p", type=float, default=0.1, help="学习率衰减指数")
    parser.add_argument("--q", type=float, default=0.6, help="步长衰减指数")
    parser.add_argument("--c0", type=float, default=1.0, help="初始步长")
    parser.add_argument("--d2_steps", type=int, default=1, help="每次迭代中判别器2(θ)的更新次数")
    parser.add_argument("--method", type=int, default=1, help="优化方法选择: 1 for gradient based, 2 for loss based")
    parser.add_argument("--result_dir", type=str, default="images", help="结果保存路径")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard日志保存路径")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="是否使用TensorBoard")
    parser.add_argument("--dataset", type=str, default="mnist", help="数据集名称")
    opt = parser.parse_args()
    
    # 初始化随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建保存图像的目录
    os.makedirs(opt.result_dir, exist_ok=True)
    
    # 创建TensorBoard日志目录
    if opt.use_tensorboard:
        os.makedirs(opt.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=opt.log_dir)
    else:
        writer = None
    
    # 使用CUDA进行训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 配置数据加载器
    if opt.dataset == "mnist":
        # MNIST数据集
        os.makedirs("data/mnist", exist_ok=True)
        transform = transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataloader = DataLoader(
            datasets.MNIST("data/mnist", train=True, download=True, transform=transform),
            batch_size=opt.batch_size,
            shuffle=True,
        )
    else:
        raise NotImplementedError(f"未实现的数据集: {opt.dataset}")
    
    # 初始化模型
    generator = Generator(opt.latent_dim, opt.img_size, opt.channels).to(device)
    discriminator1 = Discriminator(opt.img_size, opt.channels).to(device)
    discriminator2 = Discriminator(opt.img_size, opt.channels).to(device)
    print(generator)
    print(discriminator1)
    
    # 初始化优化器 (使用Adam)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=opt.d1_lr, betas=(opt.b1, opt.b2))
    optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=opt.d2_lr, betas=(opt.b1, opt.b2))
    
    # 损失记录
    losses = {"d1": [], "d2": [], "g": []}
    training_time = []

    # 开始训练
    print("开始训练...")
    
    # 初始化拉格朗日乘子
    lamk = torch.tensor(0.0, requires_grad=False, device=device)
    zk = torch.tensor(0.0, requires_grad=False, device=device)
    
    start_time = time.time()
    
    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()
        epoch_g_loss = 0
        epoch_d1_loss = 0
        epoch_d2_loss = 0
        
        for i, (imgs, _) in enumerate(dataloader):
            step = epoch * len(dataloader) + i
            
            # 配置输入
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # 计算当前迭代的学习率
            k = step + 1
            current_d1_lr = opt.d1_lr * (k ** (-opt.p))
            current_d2_lr = opt.d2_lr * (k ** (-opt.p))
            current_g_lr = opt.g_lr * (k ** (-opt.p))
            ck = opt.c0 * (k ** (-opt.q))
            
            # 更新学习率
            for param_group in optimizer_D1.param_groups:
                param_group['lr'] = current_d1_lr
            for param_group in optimizer_D2.param_groups:
                param_group['lr'] = current_d2_lr
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = current_g_lr
            
            # ---------------------
            #  更新判别器D2 (θ)
            # ---------------------
            
            # 循环多次更新D2
            d2_loss_sum = 0
            g_theta_sum = 0
            
            for _ in range(opt.d2_steps):
                optimizer_D2.zero_grad()
                
                # 生成随机噪声
                z = torch.randn(batch_size, opt.latent_dim).to(device)
                
                # 生成样本
                with torch.no_grad():
                    fake_imgs = generator(z)
                
                # 计算D2的实分数和假分数
                d2_real = discriminator2(real_imgs.detach())
                d2_fake = discriminator2(fake_imgs.detach())
                
                # Wasserstein距离
                d2_loss = torch.mean(d2_fake) - torch.mean(d2_real)
                
                # 计算梯度惩罚
                alpha = torch.rand(batch_size, 1, 1, 1).to(device)
                interpolates = alpha * real_imgs.detach() + (1 - alpha) * fake_imgs.detach()
                interpolates.requires_grad = True
                d2_interpolates = discriminator2(interpolates)
                gradients = torch.autograd.grad(
                    outputs=d2_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(d2_interpolates),
                    create_graph=True,
                    only_inputs=True,
                )[0]
                gradients = gradients.view(batch_size, -1)
                gradient_norm = gradients.norm(2, dim=1)
                g_theta = ((gradient_norm - 1) ** 2).mean()
                
                # 添加拉格朗日乘子项
                constraint_term = zk * g_theta
                
                # 近端项 - 计算D1和D2之间的参数差距
                proximal_term = 0
                for p1, p2 in zip(discriminator1.parameters(), discriminator2.parameters()):
                    p1 = p1.detach()
                    proximal_term += ((p1 - p2) ** 2).sum()
                proximal_term = proximal_term / opt.gam1
                
                # 总损失
                d2_total_loss = d2_loss + constraint_term + proximal_term
                
                # 反向传播和更新
                d2_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator2.parameters(), opt.clip_value)
                optimizer_D2.step()
                
                d2_loss_sum += d2_loss.item()
                g_theta_sum += g_theta.item()
            
            # 计算平均D2损失和梯度范数
            d2_loss_avg = d2_loss_sum / opt.d2_steps
            g_theta_avg = g_theta_sum / opt.d2_steps
            
            # ---------------------
            #  计算D1的梯度惩罚 g(x^k, y^k)
            # ---------------------
            
            # 重新生成随机样本进行D1的梯度惩罚计算
            with torch.no_grad():
                z = torch.randn(batch_size, opt.latent_dim).to(device)
                fake_imgs = generator(z)
                
            # 单独的正向传播来计算g_x_y
            alpha = torch.rand(batch_size, 1, 1, 1).to(device)
            interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs
            interpolates.requires_grad = True
            d1_interpolates = discriminator1(interpolates)
            
            # 计算D1梯度
            gradients = torch.autograd.grad(
                outputs=d1_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(d1_interpolates),
                create_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(batch_size, -1)
            gradient_norm = gradients.norm(2, dim=1)
            g_x_y = ((gradient_norm - 1) ** 2).mean()
            
            # 更新拉格朗日乘子λ和z
            lamk_new = torch.clamp(zk + opt.gam2 * g_x_y, min=0)
            d_z = -(zk - lamk_new) / opt.gam2 - g_theta_avg
            alpha_k = opt.c0 * ((step + 1) ** (-opt.q))
            zk_new = torch.clamp(zk - alpha_k * d_z, min=0)
            
            # ---------------------
            #  更新判别器D1 (y)
            # ---------------------
            
            optimizer_D1.zero_grad()
            
            # 重新生成批次
            z = torch.randn(batch_size, opt.latent_dim).to(device)
            with torch.no_grad():
                fake_imgs = generator(z)
            
            # 计算D1的实分数和假分数
            d1_real = discriminator1(real_imgs)
            d1_fake = discriminator1(fake_imgs)
            
            # Wasserstein距离
            d1_loss = torch.mean(d1_fake) - torch.mean(d1_real)
            
            # 计算梯度惩罚
            alpha = torch.rand(batch_size, 1, 1, 1).to(device)
            interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs
            interpolates.requires_grad = True
            d1_interpolates = discriminator1(interpolates)
            gradients = torch.autograd.grad(
                outputs=d1_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(d1_interpolates),
                create_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(batch_size, -1)
            gradient_norm = gradients.norm(2, dim=1)
            d1_gradient_penalty = ((gradient_norm - 1) ** 2).mean()
            
            # 添加拉格朗日乘子项
            lagrangian_term = lamk_new * d1_gradient_penalty
            
            # 总损失
            d1_total_loss = d1_loss + lagrangian_term
            
            # 反向传播和更新
            d1_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator1.parameters(), opt.clip_value)
            optimizer_D1.step()
            
            # ---------------------
            #  更新生成器G (x)
            # ---------------------
            
            optimizer_G.zero_grad()
            
            # 重新生成批次
            z = torch.randn(batch_size, opt.latent_dim).to(device)
            fake_imgs = generator(z)
            
            # G的损失 (让D1尽可能认为生成的图像是真的)
            g_loss = -torch.mean(discriminator1(fake_imgs))
            
            # 反向传播和更新
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), opt.clip_value)
            optimizer_G.step()
            
            # 更新z和λ
            lamk = lamk_new.detach()
            zk = zk_new.detach()
            
            # 更新惩罚参数
            if (step + 1) % opt.penalty_update_freq == 0:
                opt.rho1 = opt.rho1 * opt.penalty_increase_factor
                opt.rho2 = opt.rho2 * opt.penalty_increase_factor
            
            # 打印训练状态
            if (step + 1) % opt.print_freq == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D1 loss: %f] [D2 loss: %f] [G loss: %f] [λ: %f] [z: %f] [ρ1: %f] [ρ2: %f]"
                    % (epoch+1, opt.n_epochs, i+1, len(dataloader), d1_loss.item(), d2_loss_avg, g_loss.item(), lamk.item(), zk.item(), opt.rho1, opt.rho2)
                )
            
            # 记录到tensorboard
            if writer:
                writer.add_scalar("D1/loss", d1_loss.item(), step)
                writer.add_scalar("D2/loss", d2_loss_avg, step)
                writer.add_scalar("G/loss", g_loss.item(), step)
                writer.add_scalar("lambda", lamk.item(), step)
                writer.add_scalar("z", zk.item(), step)
                writer.add_scalar("penalty/rho1", opt.rho1, step)
                writer.add_scalar("penalty/rho2", opt.rho2, step)
                writer.add_scalar("lr/d1", current_d1_lr, step)
                writer.add_scalar("lr/d2", current_d2_lr, step)
                writer.add_scalar("lr/g", current_g_lr, step)
                writer.add_scalar("constraint/g_x_y", g_x_y.item(), step)
                writer.add_scalar("constraint/g_x_theta", g_theta_avg, step)
            
            # 累计损失
            epoch_d1_loss += d1_loss.item()
            epoch_d2_loss += d2_loss_avg
            epoch_g_loss += g_loss.item()
        
        # 计算平均损失
        epoch_d1_loss /= len(dataloader)
        epoch_d2_loss /= len(dataloader)
        epoch_g_loss /= len(dataloader)
        losses["d1"].append(epoch_d1_loss)
        losses["d2"].append(epoch_d2_loss)
        losses["g"].append(epoch_g_loss)
        
        # 记录训练时间
        epoch_time = time.time() - epoch_start_time
        training_time.append(time.time() - start_time)
        
        # 生成样本图像
        z = torch.randn(25, opt.latent_dim).to(device)
        with torch.no_grad():
            gen_imgs = generator(z)
        save_image(gen_imgs.data, f"{opt.result_dir}/bi_wgan_epoch_{epoch}.png", nrow=5, normalize=True)
        
        # 保存模型
        if (epoch + 1) % opt.save_freq == 0:
            save_model(epoch, generator, discriminator1, discriminator2,
                     optimizer_G, optimizer_D1, optimizer_D2, lamk, zk, opt)
        
        print(f"\nEpoch {epoch}/{opt.n_epochs} completed in {epoch_time:.2f}s. D1 loss: {epoch_d1_loss:.6f}, D2 loss: {epoch_d2_loss:.6f}, G loss: {epoch_g_loss:.6f}")
    
    # 保存最终模型
    save_model(opt.n_epochs-1, generator, discriminator1, discriminator2,
             optimizer_G, optimizer_D1, optimizer_D2, lamk, zk, opt)
    
    # 保存训练记录
    pd.DataFrame({
        "Time": training_time,
        "D1_Loss": losses["d1"],
        "D2_Loss": losses["d2"],
        "G_Loss": losses["g"]
    }).to_csv(f"{opt.result_dir}/bi_wgan_training_log.csv", index=False)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses["d1"], label="Discriminator1")
    plt.plot(losses["d2"], label="Discriminator2")
    plt.plot(losses["g"], label="Generator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{opt.result_dir}/bi_wgan_loss_curve.png")
    plt.close()
    
    # 关闭TensorBoard
    if writer:
        writer.close()
    
    print("训练完成!")

# 模型保存函数
def save_model(epoch, generator, discriminator1, discriminator2, 
               optimizer_G, optimizer_D1, optimizer_D2, lamk, zk, opt):
    """保存模型和训练状态"""
    save_path = f"{opt.result_dir}/model_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator1_state_dict': discriminator1.state_dict(),
        'discriminator2_state_dict': discriminator2.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D1_state_dict': optimizer_D1.state_dict(),
        'optimizer_D2_state_dict': optimizer_D2.state_dict(),
        'lamk': lamk,
        'zk': zk,
        'opt': opt,
    }, save_path)
    print(f"模型已保存到 {save_path}")

# 生成器模型
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.channels, self.img_size, self.img_size)
        return img

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels

        self.model = nn.Sequential(
            nn.Linear(channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

if __name__ == "__main__":
    main()
