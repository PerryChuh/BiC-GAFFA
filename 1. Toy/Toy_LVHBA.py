#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time 

from Toy_utils import Monitor

from itertools import product
from tqdm import tqdm

#%%
if True:
    t00 = time.time()
    verbose = 2
    n = 1000

    P, nIter = 1, 50000

    p = 0.3
    gam1, gam2 = 10., 1.
    alphak, betak, etak = 5e-3, 2e-3, 3e-2
    u = 1
    resultfile = f'./results/Toy_P={P}_LVHBA_gam1={gam1}_gam2={gam2}_etak={etak}_alphak={alphak}_p={p}.csv'
    

    F   = lambda x, y: (y[:n] - 2) @ (x - 1) + np.sum(np.square(y[n:] + 3))
    dxF = lambda x, y: y[:n] - 2
    dyF = lambda x, y: np.concatenate([x - 1, 2 * (y[n:] + 3)])

    f   = lambda x, y: .5 * np.sum(np.square(y[:n])) - x @ y[:n] + np.sum(y[n:])
    dxf = lambda x, y: -1 * y[:n]
    dyf = lambda x, y: np.concatenate([y[:n] - x, np.ones(n)])

    if P == 1: 
        h = lambda x : np.sum(x)
        dh = lambda x : np.ones(n)
        projxy = lambda w: w - sum(w) / len(w)
    elif P == 3:
        h = lambda x : np.sum(x ** 3)
        dh = lambda x : 3 * x**2

        import pyomo.environ as pyo
        projx3 = pyo.AbstractModel()
        projx3.N = pyo.Set(initialize = range(n * 3))
        projx3.w = pyo.Var(projx3.N, domain=pyo.Reals)
        projx3.w0 = pyo.Param(projx3.N, mutable = True)
        def obj_rule(model):
            return sum((model.w[i] - model.w0[i])**2 for i in model.N)
        projx3.obj = pyo.Objective(rule = obj_rule)
        def con_rule(model):
            return sum(model.w[i] ** 3 for i in range(n)) + sum(model.w[n+i] for i in range(2*n)) == 0 
        projx3.con = pyo.Constraint(rule = con_rule)
        projx3_model = projx3.create_instance()
        opt = pyo.SolverFactory('ipopt')
        def projxy(w0):
            for i in range(3*n):
                projx3_model.w0[i] = w0[i]
            opt.solve(projx3_model)
            return np.array([pyo.value(projx3_model.w[i]) for i in projx3_model.N])
        
    g    = lambda x, y: h(x) + np.sum(y)
    dxg  = lambda x, y: dh(x)
    dyg  = lambda x, y: np.ones(2 * n)
    feas = lambda x, y, lam: np.sqrt((lam + 1) **2 + np.sum(np.square(y[:n] - x - 1)) + (np.sum(x + 1 + y[n:]) + h(x))**2 )

    proj1 = lambda x, b : x - (sum(x) + b) / len(x)  # proj1(x) + b = 0
    yx   = lambda x, y: np.concatenate((x+1, proj1(y[n:], sum(x+1)+h(x))))
    y2yx = lambda x, y: np.linalg.norm( y - yx(x, y) )

    # opt: (1, 2, -3)
    xopt = 1 * np.ones(n)
    yopt = np.concatenate([2 * np.ones(n), -3 * np.ones(n)])

    metric_x = lambda x, y : np.linalg.norm(x - xopt) / np.linalg.norm(xopt)
    metric_y = lambda x, y : y2yx(x, y) / np.linalg.norm(yx(x, y))
    
    if verbose: print(f'Opt: F = {F(xopt, yopt):.2f}, f = {f(xopt, yopt):.2f}, g = {g(xopt, yopt):.2f}, feas = {feas(xopt, yopt, -1):.2f}')

    # initial guess
    x0 = 0 * np.ones(n)
    y0 = 1 * np.concatenate([np.ones(n), np.ones(n)])
    y0 = proj1(y0, h(x0))
    y1 = np.copy(y0)

    lam0 = -1.
    lam1 = -1.

    monitor = Monitor()
    T = 0
    monitor.append({"k": 0, "time": T, 
                "F": F(x0, y0), "f": f(x0, y0), "g": g(x0, y0), 
                "dx": metric_x(x0, y0), "dy": metric_y(x0, y0),  
                })

    if verbose: print(f'{0:5d}-th iter: F = {F(x0, y0):.2f}, f = {f(x0, y0):.2f}, g = {g(x0, y0):.2f}, feas = {feas(x0, y0, lam0):.2f}')

    for k in range(nIter):
        time0 = time.time()
        # ck = (k + 1) ** p 
        ck = (k + 1) ** p 

        dy1 = dyf(x0, y1) + lam0 * dyg(x0, y1) + 1 / gam1 * (y1 - y0) 
        dlam0 = - g(x0, y1) + 1 / gam2 * (lam0 - lam1)

        y1 = y1 - etak * dy1
        lam0 = lam0 - etak * dlam0
        lam0 = np.maximum(np.minimum(lam0, u), -u)

        dx0 = 1 / ck * dxF(x0, y0) + dxf(x0, y0) - dxf(x0, y1) - lam0 * dxg(x0, y1)
        dy0 = 1 / ck * dyF(x0, y0) + dyf(x0, y0) - 1 / gam1 * (y0 - y1)
        dlam1 = - 1 / gam2 * (lam0 - lam1)

        x0 = x0 - alphak * dx0
        y0 = y0 - alphak * dy0
        w0 = np.concatenate([x0, y0])
        w0 = projxy(w0)
        x0, y0 = w0[:n], w0[n:]
        lam1 = lam1 - betak * dlam1
        lam1 = np.maximum(np.minimum(lam1, u), -u)

        T += time.time() - time0
        monitor.append({"k": k+1, "time": T,
                        "F": F(x0, y0), "f": f(x0, y0), "g": g(x0, y0), 
                        "dx": metric_x(x0, y0), "dy": metric_y(x0, y0),  
                        })
        if np.isnan(F(x0, y0)): break
        if verbose: print(f'{k+1:5d}-th iter: F = {monitor.F[-1]:.2f}, f = {monitor.f[-1]:.2f}, g = {monitor.g[-1]:.2f}, feas = {monitor.feasibility[-1]:.2f}, lam = {lam0:.2f}')

    monitor.save_csv(resultfile)
    print(f"Total time: {time.time() - t00:.1f} s")

    if verbose>1:
        data = pd.read_csv(resultfile)
        plt.semilogy(data["time"], data["dx"], label=r'$\|\|x-x*\|\|/\|\|x^*\|\|$')
        plt.semilogy(data["time"], data["dy"], label=r'$\|\|y-y*\|\|/\|\|y^*\|\|$')
        plt.legend()
        plt.show()

# %%
