#%%
import numpy as np
import scipy as sp
import pandas as pd 
import time

from Toy_utils import Monitor
import matplotlib.pyplot as plt

from itertools import product
from tqdm import tqdm

class EAIPOD_Setting:
    def __init__(self, n = 1000, P = 1, K = 50, S = 5, T = 2, lr = 0.001, hlr = 0.0005, p = 1.0, seed=0) -> None:
        self.n = n 
        self.P = P
        self.K = K
        self.S = S
        self.T = T
        self.lr = lr
        self.hlr = hlr
        self.p = p
        self.seed = seed

#%%
def run(setting=EAIPOD_Setting(), verbose=0):
    # read parameters
    n = setting.n
    P = setting.P
    K = setting.K
    S = setting.S
    T = setting.T
    lr = setting.lr
    hlr = setting.hlr
    p = setting.p
    seed = setting.seed
    filename = f'./results/EAIPOD_P={P}_n={n}_K={K}_S={S}_T={T}_p={p}_lr={lr}_hlr={hlr}_seed={seed}.csv'

    F = lambda x, y1, y2: ((x - 1).T @ (y1 - 2)).item() + np.sum(np.square(y2 + 3))
    f = lambda x, y1, y2: .5 * np.sum(np.square(y1)) - (x.T @ y1).item() + np.sum(y2)

    if P == 1:
        h  = lambda x : np.sum(x)
        dh = lambda x : np.ones([n, 1])
    elif P == 3:
        h  = lambda x : np.sum(x ** 3)
        dh = lambda x : 3 * x ** 2
    g = lambda x, y1, y2: h(x) + np.sum(y1) + np.sum(y2)
    proj1 = lambda x, b : x - (sum(x) + b) / len(x)  # proj1(x) + b = 0
    
    yx = lambda x, y: np.concatenate((x+1, proj1(y[n:], sum(x+1)+h(x))))
    y2yx = lambda x, y: np.linalg.norm(y - np.concatenate((x+1, proj1(y[n:], sum(x+1)+h(x)))))

    e1 = np.ones((n, 1))
    e2 = np.ones((2*n, 1))
    E = np.eye(n)
    O = np.zeros((n,n))
    A = np.concatenate((E, O), axis=1)
    A_ = np.linalg.pinv(np.ones((2*n, 1)).T)

    # here f = F, g = f 
    df_dx = lambda x, y : y[:n] - 2
    df_dy = lambda x, y : np.concatenate([x - 1, 2 * (y[n:] + 3)], axis=0)
    dg_dy = lambda x, y : np.concatenate([y[:n] - x, e1], axis=0)
    dg_dxy= lambda x, y : A
    dg_dyy= lambda x, y : A.T@A

    x_opt  = e1
    x_opt_norm = np.linalg.norm(x_opt)

    yxstar = lambda x, y: np.concatenate((x+1, proj1(y[n:], sum(x+1)+h(x))))
    metric_x = lambda x, y : np.linalg.norm(x - x_opt) / x_opt_norm
    metric_y = lambda x, y : y2yx(x, y) / np.linalg.norm(yx(x, y))

    monitor = Monitor()

    np.random.seed(seed)
    x  = 0 * np.ones((n,1))
    y1 = 1 * np.ones((n,1))
    y2 = 1 * np.ones((n,1))
    y = np.concatenate((y1, y2), axis=0)
    y = proj1(y, h(x))
    Time = 0
    monitor.append({"k": 0, "time": Time,
                    "F": F(x, y[:n], y[n:]), "f": f(x, y[:n], y[n:]), 
                    "g": g(x, y[:n], y[n:]),
                    "dx": metric_x(x, y), "dy": metric_y(x, y),  
                    })
    r = np.zeros(y.shape)
    v = sp.linalg.null_space(e2.T)

    # compute metric at first round

    temp = np.linalg.pinv(v.T@dg_dyy(x, y)@v)
    w = v@temp@(v.T)@df_dy(x, y)
    hg = (dh(x) @ (A_.T) @ dg_dyy(x, y) - dg_dxy(x, y)) @ w
    dF = df_dx(x, y) + hg

    if verbose: print(f'{0:5d}-th iter: F = {monitor.F[-1]:.2f}, f = {monitor.f[-1]:.2f}, g = {monitor.g[-1]:.2f}')
    
    Kp = np.min((int(K/p), 5000))


    for k in range(Kp):
        time0 = time.time()
        for s in range(S):
            # dg_dy = A.T@A@y-A.T@x+A.T@e1
            # n = rng.normal(0,sig,dg_dy.shape) 
            z = y - lr*(dg_dy(x, y)-r)
            theta = np.random.binomial(1,p)
            if theta == 1:
                y = z - lr/p * r
                y = proj1(y, h(x))
                r = r + p/lr * (y-z)
            else:
                y = z
            # proj_count+=theta
        # this is problematic when C is not I
        # df_dy = (A.T@(A@y-e1))+B.T@(B@y+e1)
        # df_dy+=rng.normal(0,sig,df_dy.shape)
        # dg_dyy = A.T@A
        # dg_dyy+=rng.normal(0,sig,dg_dyy.shape)
        # temp = np.linalg.pinv(v.T @ dg_dyy(x, y) @ v)
        w = v@temp@(v.T)@df_dy(x, y)
        for t in range(T):
            # df_dx = np.zeros((n, 1))
            # df_dx=x-A@y
            # df_dx+=rng.normal(0,sig,df_dx.shape)
            # dh = e1
            # dh = 3 * x ** 2
            # dh+=rng.normal(0,sig,dh.shape)
            # dg_dxy = A
            # dg_dyy+=rng.normal(0,sig,dg_dxy.shape)
            # dg_dyy = A.T@A
            # dg_dyy+=rng.normal(0,sig,dg_dyy.shape)
            # hg = (dh(x) @ (A_.T) @ dg_dyy(x, y) - dg_dxy(x, y)) @ w
            dF = df_dx(x, y) + hg
            x = x - hlr*(dF)

        # v=sp.linalg.null_space(e2.T)

        Time += time.time() - time0
        y0star = yxstar(x, y)
        monitor.append({"k": k+1, "time": Time,
                        "F": F(x, y[:n], y[n:]), "f": f(x, y[:n], y[n:]), 
                        "g": g(x, y[:n], y[n:]),
                        "dx": np.linalg.norm(x - x_opt) / x_opt_norm, 
                        "dy": np.linalg.norm(y - y0star) / np.linalg.norm(y0star),  
                        })

        if np.isnan(F(x, y[:n], y[n:])): break
        # if verbose: print(f'{k+1:5d}-th iter: F = {monitor.F[-1]:.2f}, f = {monitor.f[-1]:.2f}, g = {monitor.g[-1]:.2f}')
        if verbose: print(f'{k+1:5d}-th iter: F = {monitor.F[-1]:.2f}, f = {monitor.f[-1]:.2f}, g = {monitor.g[-1]:.2f}, dx = {monitor.dx[-1]:.2f}, dy = {monitor.dy[-1]:.2f}')
    monitor.save_csv(filename)
    if verbose: print(f'end time {Time:.1f}')
    if verbose>1:
        logs_dict = pd.read_csv(filename)
        plt.plot(logs_dict["time"], logs_dict["dx"], label='||x-x*||')
        plt.plot(logs_dict["time"], logs_dict["dy"], label='||y-y*||')
        plt.title("E-AiPOD")
        plt.xlabel("Time")
        plt.show()


def show_result(setting=EAIPOD_Setting()):
    n = setting.n
    P = setting.P
    K = setting.K
    S = setting.S
    T = setting.T
    lr = setting.lr
    hlr = setting.hlr
    p = setting.p
    seed = setting.seed
    filename = f'./results/EAIPOD_P={P}_n={n}_K={K}_S={S}_T={T}_p={p}_lr={lr}_hlr={hlr}_seed={seed}.csv'

    print(setting.__dict__)

    logs_dict = pd.read_csv(filename)
    plt.plot(logs_dict["time"], logs_dict["dx"], label='||x-x*||')
    plt.plot(logs_dict["time"], logs_dict["dy"], label='||y-y*||')
    plt.title("E-AiPOD")
    plt.xlabel("Time")
    plt.show()    



#%%
if __name__ == "__main__":
    setting = EAIPOD_Setting(n = 1000, P = 1, 
        K = 50, S = 5, T = 2, 
        lr = 0.001, hlr = 0.0005, 
        p = 1.0, seed=0)
    hlrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    lrs  = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    for hlr, lr in tqdm(product(hlrs, lrs)):
        setting.hlr = hlr 
        setting.lr = lr
        setting.K = 50
        setting.P = 1
        run(P=1, setting=setting, verbose=0)
        show_result(setting)
        setting.K = 1000
        setting.P = 3
        run(P=3, setting=setting, verbose=0)
        show_result(setting)
# %%
