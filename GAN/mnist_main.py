import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from matplotlib.gridspec import GridSpec
from wgan import main as wgan_main, opts as wgan_opts
# 直接导入bi_wgan模块，因为它不提供opts类
import bi_wgan

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
        # 加载WGAN模型
        wgan_opt = wgan_opts()
        wgan_opt.update_prefix()
        wgan_path = f"{wgan_opt.prefix}_generator.pth"
        
        # 加载BiC-WGAN模型
        bic_path = "images/model_epoch_99.pth"  # 默认使用最后一个epoch的模型
        
        wgan_generator.load_state_dict(torch.load(wgan_path))
        
        # BiC-WGAN保存的是整个检查点，需要提取生成器状态
        bic_checkpoint = torch.load(bic_path)
        bic_wgan_generator.load_state_dict(bic_checkpoint['generator_state_dict'])
        
        print(f"已加载WGAN模型: {wgan_path}")
        print(f"已加载BiC-WGAN模型: {bic_path}")
    except FileNotFoundError as e:
        print(f"错误：无法找到训练好的模型文件: {e}")
        print("请先运行训练流程：python GAN/mnist_main.py --model both")
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

def compare_losses():
    """比较两种模型的训练损失曲线"""
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 加载训练记录
    try:
        wgan_log = pd.read_csv("images/wgan_training_log.csv")
        bic_wgan_log = pd.read_csv("images/bi_wgan_training_log.csv")
    except FileNotFoundError:
        print("错误：无法找到训练记录文件。请先运行训练流程：python GAN/mnist_main.py --model both")
        return
    
    # 创建图表
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 绘制判别器损失
    axs[0].plot(wgan_log['D_Loss'], label='WGAN-D', linewidth=2)
    axs[0].plot(bic_wgan_log['D1_Loss'], label='BiC-WGAN-D1', linewidth=2)
    axs[0].plot(bic_wgan_log['D2_Loss'], label='BiC-WGAN-D2', linewidth=2)
    axs[0].set_ylabel('Discriminator loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title('WGAN vs BiC-WGAN Discriminator Loss Comparison')
    
    # 绘制生成器损失
    axs[1].plot(wgan_log['G_Loss'], label='WGAN-G', color='green', linewidth=2)
    axs[1].plot(bic_wgan_log['G_Loss'], label='BiC-WGAN-G', color='purple', linewidth=2)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Generator loss')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    axs[1].set_title('WGAN vs BiC-WGAN Generator Loss Comparison')
    
    plt.tight_layout()
    plt.savefig("results/loss_comparison.png", dpi=300)
    plt.close()
    
    print("损失曲线比较已保存到 results/loss_comparison.png")

def show_generation_progress(model="both", epochs=None):
    """显示生成图像质量随训练过程的变化"""
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 如果没有指定epochs，则默认每5个epoch取一个点
    if epochs is None:
        epochs = list(range(0, 101, 5))
    
    model_names = []
    if model == "wgan" or model == "both":
        model_names.append("wgan")
    if model == "bic_wgan" or model == "both":
        model_names.append("bi_wgan")
    
    for model_name in model_names:
        # 获取图像文件路径
        image_paths = [f"images/{model_name}_epoch_{epoch}.png" for epoch in epochs]
        
        # 检查文件是否存在
        missing_files = [path for path in image_paths if not os.path.exists(path)]
        if missing_files:
            print(f"警告：无法找到以下文件: {missing_files}")
            print(f"请确认指定的epoch已运行，或者修改epochs参数")
            continue
        
        # 创建图表 - 根据epoch数量动态调整图像大小
        fig_width = min(24, len(epochs) * 1.2)  # 限制最大宽度
        fig = plt.figure(figsize=(fig_width, 5))
        
        # 添加标题
        fig.suptitle(f"{model_name.upper()} 生成图像随训练进度的变化 (每5个Epoch)", fontsize=16)
        
        # 添加子图
        for i, epoch in enumerate(epochs):
            img_path = image_paths[i]
            img = plt.imread(img_path)
            
            ax = plt.subplot(1, len(epochs), i+1)
            ax.imshow(img)
            ax.set_title(f"Epoch {epoch}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_progress_sequence.png", dpi=300)
        plt.close()
        
        print(f"{model_name} 模型生成进度序列已保存到 results/{model_name}_progress_sequence.png")
    
    # 如果两个模型都要比较，创建单独的对比图
    if model == "both":
        # 使用网格子图，根据epoch数量动态调整图像大小
        fig_width = min(24, len(epochs) * 1.2)  # 限制最大宽度
        fig = plt.figure(figsize=(fig_width, 10))
        gs = GridSpec(2, len(epochs), figure=fig)
        
        fig.suptitle("WGAN vs BiC-WGAN 生成图像质量随训练进度的序列对比 (每5个Epoch)", fontsize=16)
        
        for i, epoch in enumerate(epochs):
            # WGAN
            wgan_img_path = f"images/wgan_epoch_{epoch}.png"
            if os.path.exists(wgan_img_path):
                wgan_img = plt.imread(wgan_img_path)
                ax1 = fig.add_subplot(gs[0, i])
                ax1.imshow(wgan_img)
                ax1.set_title(f"Epoch {epoch}", fontsize=10)
                ax1.axis('off')
                if i == 0:
                    ax1.set_ylabel("WGAN", fontsize=14)
            
            # BiC-WGAN
            bic_wgan_img_path = f"images/bi_wgan_epoch_{epoch}.png"
            if os.path.exists(bic_wgan_img_path):
                bic_wgan_img = plt.imread(bic_wgan_img_path)
                ax2 = fig.add_subplot(gs[1, i])
                ax2.imshow(bic_wgan_img)
                ax2.set_title(f"Epoch {epoch}", fontsize=10)
                ax2.axis('off')
                if i == 0:
                    ax2.set_ylabel("BiC-WGAN", fontsize=14)
        
        plt.tight_layout()
        plt.savefig("results/model_comparison_sequence.png", dpi=300)
        plt.close()
        
        print("两种模型生成质量序列对比已保存到 results/model_comparison_sequence.png")

def main():
    parser = argparse.ArgumentParser(description="MNIST手写数字生成实验")
    parser.add_argument("--model", type=str, default="both", choices=["wgan", "bic_wgan", "both", "compare"],
                        help="选择要运行的模型: wgan, bic_wgan, both, 或 compare")
    # 共享参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--latent_dim", type=int, default=100, help="潜在空间维度")
    parser.add_argument("--sample_interval", type=int, default=1, help="保存样本间隔，默认为1表示每个epoch都保存")
    
    # WGAN特定参数
    parser.add_argument("--wgan_d_lr", type=float, default=0.0002, help="WGAN判别器学习率")
    parser.add_argument("--wgan_g_lr", type=float, default=0.0002, help="WGAN生成器学习率")
    parser.add_argument("--wgan_clip_value", type=float, default=0.01, help="WGAN梯度裁剪值")
    parser.add_argument("--wgan_d_steps", type=int, default=5, help="WGAN每轮更新判别器的次数")
    
    # BiC-WGAN特定参数
    parser.add_argument("--bic_d1_lr", type=float, default=0.0002, help="BiC-WGAN判别器1学习率")
    parser.add_argument("--bic_d2_lr", type=float, default=0.0001, help="BiC-WGAN判别器2学习率")
    parser.add_argument("--bic_g_lr", type=float, default=0.0002, help="BiC-WGAN生成器学习率")
    parser.add_argument("--bic_clip_value", type=float, default=1.0, help="BiC-WGAN梯度裁剪值")
    parser.add_argument("--bic_d2_steps", type=int, default=1, help="BiC-WGAN每轮更新判别器2的次数")
    parser.add_argument("--bic_gam1", type=float, default=0.5, help="BiC-WGAN判别器1和判别器2之间的正则化系数")
    parser.add_argument("--bic_gam2", type=float, default=0.5, help="BiC-WGAN拉格朗日乘子正则化系数")
    parser.add_argument("--bic_gradient_penalty", type=float, default=10, help="BiC-WGAN梯度惩罚系数")
    
    # 可视化分析参数
    parser.add_argument("--analysis", type=str, default="all", choices=["losses", "progress", "compare_models", "all"],
                        help="选择分析类型: losses (损失对比), progress (生成质量变化), compare_models (模型对比), all (全部)")
    parser.add_argument("--progress_epochs", type=str, default="0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99",
                        help="用于显示生成进度的epoch列表，以逗号分隔，例如'0,5,10,15,20'")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 训练模型
    if args.model in ["wgan", "both"]:
        print("开始训练WGAN模型...")
        wgan_opt = wgan_opts()
        # 设置WGAN参数
        wgan_opt.n_epochs = args.epochs
        wgan_opt.batch_size = args.batch_size
        wgan_opt.seed = args.seed
        wgan_opt.sample_interval = args.sample_interval
        wgan_opt.latent_dim = args.latent_dim
        wgan_opt.d_lr = args.wgan_d_lr
        wgan_opt.g_lr = args.wgan_g_lr
        wgan_opt.clip_value = args.wgan_clip_value
        wgan_opt.d_steps = args.wgan_d_steps
        wgan_opt.update_prefix()
        wgan_main(wgan_opt)
        print("WGAN模型训练完成!")
    
    if args.model in ["bic_wgan", "both"]:
        print("开始训练BiC-WGAN模型...")
        
        # 备份当前的命令行参数
        old_argv = sys.argv.copy()
        
        # 创建BiC-WGAN的命令行参数
        bic_argv = [
            "bi_wgan.py",
            f"--n_epochs={args.epochs}",
            f"--batch_size={args.batch_size}",
            f"--d1_lr={args.bic_d1_lr}",
            f"--d2_lr={args.bic_d2_lr}",
            f"--g_lr={args.bic_g_lr}",
            f"--latent_dim={args.latent_dim}",
            f"--sample_interval={args.sample_interval}",
            f"--clip_value={args.bic_clip_value}",
            f"--d2_steps={args.bic_d2_steps}",
            f"--gam1={args.bic_gam1}",
            f"--gam2={args.bic_gam2}",
            f"--gradient_penalty={args.bic_gradient_penalty}",
            "--result_dir=images"
        ]
        
        # 替换命令行参数
        sys.argv = bic_argv
        
        # 调用bi_wgan的main函数
        try:
            bi_wgan.main()
        finally:
            # 恢复原始命令行参数
            sys.argv = old_argv
        
        print("BiC-WGAN模型训练完成!")
    
    # 解析进度epoch列表
    progress_epochs = [int(e) for e in args.progress_epochs.split(',')]
    
    # 运行分析
    if args.model == "compare" or args.model == "both":
        if args.analysis in ["losses", "all"]:
            print("比较两种模型的训练损失...")
            compare_losses()
        
        if args.analysis in ["progress", "all"]:
            print("显示生成图像质量的变化...")
            show_generation_progress(model="both", epochs=progress_epochs)
        
        if args.analysis in ["compare_models", "all"]:
            print("比较两种模型的生成结果...")
            compare_models(samples=5, latent_dim=args.latent_dim)

if __name__ == "__main__":
    main() 