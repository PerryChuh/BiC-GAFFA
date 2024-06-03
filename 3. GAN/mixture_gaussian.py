#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.distributions import Normal, MixtureSameFamily, Categorical, Independent

#%%
def points_on_circle(num, r):
    pi = 3.141592
    coords = np.array([ ( r * np.cos((2. / num) * i * pi)  , r * np.sin((2. / num) * i * pi) ) for i in range(num)])
    return coords

def GMM(p = None):
    # Define the parameters of the mixture model
    num_mixtures = len(p)
    if num_mixtures == 25:
        means = [[-2 + i, -2+j] for i in range(5) for j in range(5)]
        means = torch.tensor(means, dtype=torch.float32)
        std = 0.05 
    elif num_mixtures == 8:
        r = 2
        coords = points_on_circle(num_mixtures, r)
        means = torch.tensor(coords, dtype=torch.float32)
        std = 0.02
        # scale = torch.tensor([[std, std] for _ in range(num_mixtures)], dtype=torch.float32)
    else:
        raise ValueError("num_mixtures must be 8 or 25")
    scale = torch.tensor([[std] for _ in range(num_mixtures)], dtype=torch.float32)
    # Create the mixture model
    coe = torch.ones(num_mixtures,)
    if p is not None:
        for i in range(num_mixtures):
            coe[i] = p[i]
    mix = Categorical(coe)
    components = Independent(Normal(means, scale), 1)
    mixture = MixtureSameFamily(mix, components)
    return mixture

#%%
def dis_draw(data, save=None):
    fig = plt.figure(figsize=(6, 4))
    plt.clf()
    sns.color_palette('Greens', n_colors=256)[0]
    sns.kdeplot(x = data[:, 0], y = data[:, 1], fill=True, cmap='Greens', n_levels=40)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    if save:
        plt.savefig(save)
    else:
        plt.show()

def sample_plot(points, title = ""):
    centers = np.array(points_on_circle(8, 2))
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.show()
    plt.close()

def show_points(points):
    plt.scatter(points[:, 0], points[:, 1], c=[0.3 for i in range(1000)], alpha=0.5)
    plt.show()
    plt.close()

def plot_samples(samples, ylabel="GAN", sample_interval = 300, savename=None):
    xmax = 5
    cols = len(samples)
    plt.figure(figsize=(2*cols, 2))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i+1, sharex=ax, sharey=ax)
        ax2 = sns.kdeplot(x = samps[:, 0], y = samps[:, 1], fill=True, cmap='Greens', n_levels=20, clip=[[-xmax,xmax]]*2)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.xticks([])
        plt.yticks([])
        plt.title('step %d'%(i*sample_interval))
    
    ax.set_ylabel(ylabel)
    plt.gcf().tight_layout()
    if savename is not None:
        plt.savefig(savename)
    plt.show()
    plt.close()

#%%
def main():
    # gen = data_generator()
    seed = 2
    rng = np.random.default_rng(seed)
    # P = rng.random(8)
    P = np.ones(8)
    P /= np.sum(P)
    gen = GMM(P)
    sample_points = gen.sample((1000,))
    # show_points(sample_points)
    dis_draw(sample_points)

if __name__ == '__main__':
    main()

# %%
