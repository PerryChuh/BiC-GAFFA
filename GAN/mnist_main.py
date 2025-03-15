import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from wgan import main as wgan_main, opts as wgan_opts
from bi_wgan import main as bic_wgan_main, opts as bic_wgan_opts

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compare_models(samples=5, n_row=5, latent_dim=100):
    """比较两种模型生成的图像"""
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 加载训练好的模型
    wgan_opt = wgan_opts()
    bic_wgan_opt = bic_wgan_opts()
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("注意：未检测到GPU，将使用CPU进行训练，这可能会较慢")
    
    # 从models.py导入模型
    from models import MNISTGenerator
    
    # 初始化模型
    wgan_generator = MNISTGenerator(latent_dim=latent_dim)
    bic_wgan_generator = MNISTGenerator(latent_dim=latent_dim)
    
    # 加载模型参数
    try:
        wgan_generator.load_state_dict(torch.load(f"{wgan_opt.prefix}_generator.pth"))
        bic_wgan_generator.load_state_dict(torch.load(f"{bic_wgan_opt.prefix}_generator.pth"))
    except FileNotFoundError:
        print("错误：无法找到训练好的模型文件。请先运行训练流程：python GAN/mnist_main.py --model both")
        return
    
    # 将模型移至设备
    wgan_generator.to(device)
    bic_wgan_generator.to(device)
    
    # 设置为评估模式
    wgan_generator.eval()
    bic_wgan_generator.eval()
    
    # 生成比较图像
    fig, axes = plt.subplots(2, samples, figsize=(samples*2, 4))
    
    # 设置标题
    fig.suptitle("WGAN vs BiC-WGAN Generated MNIST Digits", fontsize=16)
    axes[0, 0].set_ylabel("WGAN", fontsize=14)
    axes[1, 0].set_ylabel("BiC-WGAN", fontsize=14)
    
    # 对每个样本
    for i in range(samples):
        # 使用相同的随机噪声
        z = torch.randn(1, latent_dim).to(device)
        
        # 生成图像
        with torch.no_grad():
            wgan_img = wgan_generator(z).detach().cpu()
            bic_wgan_img = bic_wgan_generator(z).detach().cpu()
        
        # 显示图像
        axes[0, i].imshow(wgan_img[0, 0], cmap='gray')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(bic_wgan_img[0, 0], cmap='gray')
        axes[1, i].axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=300)
    plt.close()
    
    print("比较图像已保存到 results/comparison.png")

def main():
    parser = argparse.ArgumentParser(description="MNIST手写数字生成实验")
    parser.add_argument("--model", type=str, default="both", choices=["wgan", "bic_wgan", "both", "compare"],
                        help="选择要运行的模型: wgan, bic_wgan, both, 或 compare")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--sample_interval", type=int, default=1, help="保存样本间隔，默认为1表示每个epoch都保存")
    parser.add_argument("--latent_dim", type=int, default=100, help="潜在空间维度")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    if args.model == "wgan" or args.model == "both":
        print("开始训练WGAN模型...")
        wgan_opt = wgan_opts()
        wgan_opt.n_epochs = args.epochs
        wgan_opt.batch_size = args.batch_size
        wgan_opt.seed = args.seed
        wgan_opt.sample_interval = args.sample_interval
        wgan_opt.latent_dim = args.latent_dim
        wgan_opt.update_prefix()
        wgan_main(wgan_opt)
        print("WGAN模型训练完成!")
    
    if args.model == "bic_wgan" or args.model == "both":
        print("开始训练BiC-WGAN模型...")
        bic_wgan_opt = bic_wgan_opts()
        bic_wgan_opt.n_epochs = args.epochs
        bic_wgan_opt.batch_size = args.batch_size
        bic_wgan_opt.seed = args.seed
        bic_wgan_opt.sample_interval = args.sample_interval
        bic_wgan_opt.latent_dim = args.latent_dim
        bic_wgan_opt.update_prefix()
        bic_wgan_main(bic_wgan_opt)
        print("BiC-WGAN模型训练完成!")
    
    if args.model == "compare":
        print("比较两种模型的生成结果...")
        compare_models(samples=5, latent_dim=args.latent_dim)

if __name__ == "__main__":
    main() 