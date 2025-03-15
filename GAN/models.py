#%%
import torch
import torch.nn as nn
import numpy as np

# 用于MNIST的生成器
class MNISTGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(MNISTGenerator, self).__init__()
        self.img_shape = img_shape

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
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# 用于MNIST的判别器
class MNISTDiscriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(MNISTDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# 原始的Generator0类，保留以防需要
class Generator0(nn.Module):
    # original
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator0, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = torch.relu
    
    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        # x = self.activation_fn(self.map3(x))
        return self.map3(x)
    
# 原始的Discriminator0类，保留以防需要
class Discriminator0(nn.Module):
    # original
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator0, self).__init__()

        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = torch.relu
    
    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return torch.sigmoid(self.map3(x))

# 这些类不在本次MNIST实验中使用，可以删除
# class DiscriminatorR(nn.Module):
# ...

# WGAN的生成器
class GeneratorW(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GeneratorW, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = torch.relu
    
    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        # x = self.activation_fn(self.map3(x))
        return self.map3(x)
    
# WGAN的判别器
class DiscriminatorW(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DiscriminatorW, self).__init__()

        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = torch.relu
    
    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.map3(x)
