#%%
from gan import main as gan_main
from gan import opts as gan_opt

from wgan import main as wgan_main
from wgan import opts as wgan_opt

from wgan_gp import main as wgan_gp_main
from wgan_gp import opts as wgan_gp_opt

from con_gan import main as con_gan_main
from con_gan import opts as con_gan_opt

from gan_unroll import main as gan_unroll_main
from gan_unroll import opts as gan_unroll_opt

from bi_gan import main as bi_gan_main
from bi_gan import opts as bi_gan_opt

from bi_wgan import main as bi_wgan_main
from bi_wgan import opts as bi_wgan_opt

from bi_congan import main as bi_congan_main
from bi_congan import opts as bi_congan_opt

import time 
import os 
import pandas as pd

#%%
def main():
    gpuid = 0
    nRepeat = 5
    nRepeat = 20

    opt_gan = gan_opt()      # 1
    opt_wgan = wgan_opt()     # 2
    opt_wgan_gp = wgan_gp_opt()     # 3
    opt_con_gan = con_gan_opt()     # 4
    opt_gan_unroll = gan_unroll_opt()     # 5
    opt_bi_gan = bi_gan_opt()       # 6
    opt_bi_wgan = bi_wgan_opt()     # 7
    opt_bi_congan = bi_congan_opt()      # 8

    gan_list = [gan_main, wgan_main, wgan_gp_main, con_gan_main, gan_unroll_main, bi_gan_main, bi_wgan_main, bi_congan_main]
    opt_list = [opt_gan, opt_wgan, opt_wgan_gp, opt_con_gan, opt_gan_unroll, opt_bi_gan, opt_bi_wgan, opt_bi_congan]

    inds = [6, 7, 98]
    gans = [gan_list[i-1] for i in inds]
    opts = [opt_list[i-1] for i in inds]

    for opt in opts:
        opt.gpu = gpuid

    nG = 25
    for opt in opts:
        opt.nGaussians = nG

    t0 = time.time()
    for seed in range(nRepeat):
        print("processing seed: ", seed)
        for (gan, opt) in zip(gans, opts):
            opt.seed = seed
            gan(opt)
    print(f"Time taken: {time.time() - t0 : .1f}")

    nG = 8
    for opt in opts:
        opt.nGaussians = nG

    t0 = time.time()
    for seed in range(nRepeat):
        print("processing seed: ", seed)
        for (gan, opt) in zip(gans, opts):
            opt.seed = seed
            gan(opt)

    print(f"Time taken: {time.time() - t0 : .1f}")


# %%
if __name__ == '__main__':
    main()

