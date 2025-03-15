#%%
import geomloss
import numpy as np
import torch

p = 2
entreg = .1 # entropy regularization factor for Sinkhorn
OTLoss = geomloss.SamplesLoss(
    loss='sinkhorn', p=p,
    cost= geomloss.utils.squared_distances,
    blur=entreg**(1/p), backend='tensorized')

def OT_err(generator, gmm=None, z_dim=2, device=torch.device("cpu"), N=1000, long=False):
    if long:
        z = np.random.normal(size = (N, z_dim))
    else:
        z = np.random.normal(size = (N, z_dim)).astype(np.float32)
    z = torch.from_numpy(z).to(device)
    gen_imgs = generator(z)
    real_imgs = gmm.sample((N,)).to(device)
    return OTLoss(gen_imgs, real_imgs).item()
