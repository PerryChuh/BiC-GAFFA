#%%
import torch
import torch.nn as nn

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

class DiscriminatorR(nn.Module):
    # for unrolled
    def __init__(self, input_size, hidden_size, output_size):
        super(DiscriminatorR, self).__init__()

        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = torch.relu
    
    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return torch.sigmoid(self.map3(x))

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()


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
