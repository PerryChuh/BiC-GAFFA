#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=64, data_path='./data'):
    """
    加载MNIST数据集
    """
    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
    ])
    
    # 加载数据集
    dataset = datasets.MNIST(
        root=data_path, 
        train=True, 
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader

def save_images(generator, latent_dim, epoch, n_row=10, figsize=(10, 10), save_path=None):
    """
    生成并保存MNIST图像
    """
    # 生成图像
    z = torch.randn(n_row**2, latent_dim).to(generator.model[0].weight.device)
    gen_imgs = generator(z).detach().cpu()
    
    # 创建图像网格
    fig, axs = plt.subplots(n_row, n_row, figsize=figsize)
    
    # 逐一显示图像
    for i in range(n_row):
        for j in range(n_row):
            idx = i * n_row + j
            axs[i, j].imshow(gen_imgs[idx, 0], cmap='gray')
            axs[i, j].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(f"{save_path}.png")
    
    plt.close()
    
    return fig
