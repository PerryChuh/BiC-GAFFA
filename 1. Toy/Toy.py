#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time

from Toy_utils import Monitor, GAFFA_Setting

from itertools import product
from tqdm import tqdm
import os, sys
#%%
def run(setting = GAFFA_Setting(), verbose=0, overwrite=False):
    # read parameters
    n = setting.n
    P = setting.P
    nIter = setting.nIter
    p = setting.p 
    gam1 = setting.gam1 
    gam2 = setting.gam2 
    etak = setting.etak 
    betak = setting.betak 
    alphak = setting.alphak 
    eps = setting.eps 
    R = setting.R
    ck0 = setting.ck0
    resultfile = f'./results/Toy_P={P}_n={n}_gam1={gam1}_gam2={gam2}_etak={etak}_alphak={alphak}_p={p}.csv'
    if os.path.exists(resultfile):
        if overwrite:
            return None

    F = lambda x, y: (y[:n] - 2) @ (x - 1) + np.sum(np.square(y[n:] + 3))
    dxF = lambda x, y: y[:n] - 2
    dyF = lambda x, y: np.concatenate([x - 1, 2 * (y[n:] + 3)])

    f = lambda x, y: .5 * np.sum(np.square(y[:n])) - x @ y[:n] + np.sum(y[n:])
    dxf = lambda x, y: -1 * y[:n]
    dyf = lambda x, y: np.concatenate([y[:n] - x, np.ones(n)])

    if P == 1: 
        h = lambda x : np.sum(x)
        dh = lambda x : np.ones(n)
    elif P == 3:
        h = lambda x : np.sum(x ** 3)
        dh = lambda x : 3 * x**2
    g = lambda x, y: h(x) + np.sum(y)
    dxg = lambda x, y: dh(x)
    dyg = lambda x, y: np.ones(2 * n)

    feas = lambda x, y, lam: np.sqrt((lam + 1) **2 + np.sum(np.square(y[:n] - x - 1)) + (np.sum(x + 1 + y[n:]) + h(x))**2 )

    proj1 = lambda x, b : x - (sum(x) + b) / len(x)  # proj1(x) + b = 0
    yxstar = lambda x, y: np.concatenate((x+1, proj1(y[n:], sum(x+1)+h(x))))
    y2yx = lambda x, y: np.linalg.norm(y - np.concatenate((x+1, proj1(y[n:], sum(x+1)+h(x))))) 

    # opt: (1, 2, -3)
    xopt = 1 * np.ones(n)
    xopt_norm = np.linalg.norm(xopt)
    # yopt = np.concatenate([2 * np.ones(n), -3 * np.ones(n)])

    x0 = 0 * np.ones(n)
    y0 = 1 * np.concatenate([np.ones(n), np.ones(n)])
    
    y0 = proj1(y0, h(x0))
    y1 = np.copy(y0)

    lam0 = -1.
    lam1 = -1.

    monitor = Monitor()

    # if verbose: print(f'{0:5d}-th iter: F = {F(x0, y0):.2f}, f = {f(x0, y0):.2f}, g = {g(x0, y0):.2f}, feas = {feas(x0, y0, lam0):.2f}')

    T = 0
    y0star = yxstar(x0, y0)
    monitor.append({"k": 0, "time": T, 
                    "F": F(x0, y0), "f": f(x0, y0), "g": g(x0, y0), 
                    "dx": np.linalg.norm(x0 - xopt) / xopt_norm,
                    "dy": y2yx(x0, y0) / (np.linalg.norm(y0star)),  
                    })

    for k in range(nIter):
        time0 = time.time()
        ck = np.min([ck0 * (k + 1) ** p, 10])

        dy1 = dyf(x0, y1) + lam1 * dyg(x0, y1) + 1 / gam1 * (y1 - y0)
        y1p = y1 - etak * dy1

        lam0 = lam1 + gam2 * g(x0, y0)

        dx = 1 / ck * dxF(x0, y0) + dxf(x0, y0) + lam0 * dxg(x0, y0) - dxf(x0, y1p) - lam1 * dxg(x0, y1p) 
        x0p = x0 - alphak * dx

        dy0 = 1 / ck * dyF(x0, y0) + dyf(x0, y0) + lam0 * dyg(x0, y0) - 1 / gam1 * (y0 - y1p)
        y0 = y0 - alphak * dy0
        # y0 = proj1(y0, h(x0p))

        dlam1 = -(lam1 - lam0) / gam2 - g(x0, y1p)
        lam1 = np.minimum(np.maximum(lam1 - betak * dlam1, -R), R)

        x0, y1 = x0p, y1p

        T += time.time() - time0
        y0star = yxstar(x0, y0)
        monitor.append({"k": k+1, "time": T,
                        "F": F(x0, y0), "f": f(x0, y0), "g": g(x0, y0), 
                        "dx": np.linalg.norm(x0 - xopt) / xopt_norm, 
                        "dy": np.linalg.norm(y0 - y0star) / (np.linalg.norm(y0star)),  
                        })

        if np.isnan(F(x0, y0)): break
        if verbose: print(f'{k+1:5d}-th iter: F = {monitor.F[-1]:.2f}, f = {monitor.f[-1]:.2f}, g = {monitor.g[-1]:.2f}, feas = {monitor.feasibility[-1]:.2f}, lam = {lam0:.2f}')
        if verbose: 
            sys.stdout.write(f'{k+1:8d}-th iter: time = {T:.2f}, dx = {monitor.dx[-1]:.3f}, dy = {monitor.dy[-1]:.3f}\r')
            sys.stdout.flush()

        if max(x0) > 1e5: 
            print("x overflow")
            break 
        if monitor.dx[-1] < eps:
            break

    print(f'{k+1:8d}-th iter: time = {T:.4f}, dx = {monitor.dx[-1]:.3f}\r')
    monitor.save_csv(resultfile)
    # print(time.time() - t00)

    if verbose>1:
        logs_dict = pd.read_csv(resultfile)
        plt.semilogy(logs_dict["time"], logs_dict["dx"], label='||x-x*|| / ||x^*||')
        plt.semilogy(logs_dict["time"], logs_dict["dy"], label='||y-y*(x)|| / ||y^*(x)||')
        plt.legend()
        plt.show()

def plot_logs(setting = GAFFA_Setting()):
    n = setting.n
    P = setting.P
    nIter = setting.nIter
    p = setting.p 
    gam1 = setting.gam1 
    gam2 = setting.gam2 
    etak = setting.etak 
    betak = setting.betak 
    alphak = setting.alphak 
    eps = setting.eps 
    R = setting.R
    ck0 = setting.ck0
    resultfile = f'./results/Toy_P={P}_n={n}_gam1={gam1}_gam2={gam2}_etak={etak}_alphak={alphak}_p={p}.csv'

    logs_dict = pd.read_csv(resultfile)
    if len(logs_dict) < 10:
        return None 
    else:
        plt.semilogy(logs_dict["time"], logs_dict["dx"], label='||x-x*|| / ||x^*||')
        plt.semilogy(logs_dict["time"], logs_dict["dy"], label='||y-y*(x)|| / ||y^*(x)||')
        plt.legend()


#%%
def main0():
    # generate data in Figure. 1.
    setting = GAFFA_Setting(P = 1, n = 100, p = 0.3, nIter=500000, gam1 = 10., gam2 = 1., etak=0.01, betak=0.001, alphak = 0.001)
      
    setting.P = 1
    print(setting.__dict__)
    run(setting=setting, verbose=0)
    setting.P = 3
    print(setting.__dict__)
    run(setting=setting, verbose=0)

def main1():
    # generate data in Table. 1. 4. 5.
    ps = [.1, .2, .3, .4, .49]
    gam1s = np.array([1., 3., 5., 7., 10.])
    gam2s = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
    steps1 = [5e-4, 7e-4, 1e-3, 3e-3, 5e-3]
    steps3 = [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]
    
    for p in ps:
        setting = GAFFA_Setting(P = 1, n = 100, p = 0.3, nIter=500000, gam1 = 10., gam2 = 1., etak=0.01, betak=0.001, alphak = 0.001)
        setting.p = p

        setting.P = 1
        print(setting.__dict__)
        run(setting=setting, verbose=0)
        setting.P = 3
        print(setting.__dict__)
        run(setting=setting, verbose=0)

    for gam1 in gam1s:
        setting = GAFFA_Setting(P = 1, n = 100, p = 0.3, nIter=500000, gam1 = 10., gam2 = 1., etak=0.01, betak=0.001, alphak = 0.001)
        setting.gam1 = gam1

        setting.P = 1
        print(setting.__dict__)
        run(setting=setting, verbose=0)
        setting.P = 3
        print(setting.__dict__)
        run(setting=setting, verbose=0)

    for gam2 in gam2s:
        setting = GAFFA_Setting(P = 1, n = 100, p = 0.3, nIter=500000, gam1 = 10., gam2 = 1., etak=0.01, betak=0.001, alphak = 0.001)
        setting.gam2 = gam2
        setting.P = 1
        print(setting.__dict__)
        run(setting=setting, verbose=0)
        setting.P = 3
        print(setting.__dict__)
        run(setting=setting, verbose=0)

    for step in steps1:
        setting = GAFFA_Setting(P = 1, n = 100, p = 0.3, nIter=500000, gam1 = 10., gam2 = 1., etak=0.01, betak=0.001, alphak = 0.001)
        setting.etak = step * 10
        setting.betak = step 
        setting.alphak = step   
        setting.P = 1
        print(setting.__dict__)
        run(setting=setting, verbose=0)

    for step in steps3:
        setting = GAFFA_Setting(P = 1, n = 100, p = 0.3, nIter=500000, gam1 = 10., gam2 = 1., etak=0.01, betak=0.001, alphak = 0.001)
        setting.etak = step * 10
        setting.betak = step 
        setting.alphak = step   
        setting.P = 3
        print(setting.__dict__)
        run(setting=setting, verbose=0)

def main2():
    # generate the data in Figure. 2.
    setting = GAFFA_Setting(P = 1, n = 100, p = 0.3, nIter= 1000000, gam1 = 1., gam2 = .1, etak=5e-3, betak=5e-4, alphak = 5e-4, eps = 0.01)
    ns = [100, 300, 500, 1000, 3000, 5000, 10000, 30000]
    for n in ns:
        setting.n = n
        setting.P = 1
        stepsize = 20 / n 
        setting.etak = stepsize 
        setting.alphak = stepsize / 10 
        print(setting.__dict__)
        run(setting=setting, verbose=1, overwrite=False)
        setting.P = 3
        stepsize = 10 / n 
        setting.etak = stepsize 
        setting.alphak = stepsize / 10 
        print(setting.__dict__)
        run(setting=setting, verbose=1, overwrite=False)
    

# %%
if __name__ == "__main__":
    main0()
    # main1()
    # main2()
# %%
