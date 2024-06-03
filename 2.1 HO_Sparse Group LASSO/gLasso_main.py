#%%
import os
import numpy as np 
import cvxpy as cp
import pandas as pd
import time 
import matplotlib.pyplot as plt
from hyperopt import tpe, hp, fmin
from itertools import product
from SGL_Algorithms import Imlicit_Differntiation, iP_DCA
from gLasso_GAFFA import run as GAFFA

#%%
class Data():
    def __init__(self):
        self.X_train = 0
        self.X_validate = 0
        self.X_test = 0
        self.y_train = 0
        self.y_validate = 0
        self.y_test = 0

class Data_with_Info():
    def __init__(self, data, settings, data_index = 0) -> None:
        self.data = data 
        self.settings = settings 
        self.data_index = data_index

class SGL_Setting:
    def __init__(self, num_train=50, num_validate=50, num_test=300, num_features=150, num_experiment_groups=15):
        self.num_train = num_train
        self.num_validate = num_validate
        self.num_test = num_test
        self.num_features = num_features
        self.num_experiment_groups = num_experiment_groups
        self.num_true_groups = 5
        # num_true_groups is defined in data_generator

def load_data(root, nTr, nVal, nTest, nFeatures, nGroup, k, normalized=True):
    data = Data()
    if normalized:
        dic = np.load(root + f"data_scaled_nTr={nTr}_nVal={nVal}_nTest={nTest}_nFeatures={nFeatures}_nGroup={nGroup}_k={k}.npy", allow_pickle=True).item()
    else:
        dic = np.load(root + f"data_nTr={nTr}_nVal={nVal}_nTest={nTest}_nFeatures={nFeatures}_nGroup={nGroup}_k={k}.npy", allow_pickle=True).item()
    data.__dict__ = dic 
    return data

# collect results
class Monitor():
    def __init__(self) -> None:
        self.time = []
        self.Tr = []
        self.Val =[]
        self.Test = []
        self.Tr2 = []
        self.Val2 = []
        self.Test2 = []
        self.g = []
        self.k = []
        self.time = []
        self.dy = []
    
    def append(self, data_dic):
        for attr in self.__dict__.keys():
            self.append_one(data_dic, attr)
    
    def append_one(self, data_dic, attr):
        if attr in data_dic.keys():
            self.__dict__[attr].append(data_dic[attr])
        else:
            self.__dict__[attr].append(0)

    def to_df(self):
        return pd.DataFrame(self.__dict__)

#%%
class Training_model0:
    def __init__(self, data, settings) -> None:  
        n, p, M = settings.num_train, settings.num_features, settings.num_experiment_groups
        cal_group_sizes = [p//M] * M
        group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

        self.x = cp.Variable(p)
        self.lam = cp.Parameter(M+1, nonneg=True)
        LS_lower = .5/n * cp.sum_squares( data.y_train - data.X_train @ self.x )
        group_lasso_peanlty = [ self.lam[i] * cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) for i in range(M)]
        sparsity_penalty = self.lam[M] * cp.pnorm(self.x, 1)
        self.penalty = [cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) for i in range(M)] + [cp.pnorm(self.x, 1)]
        self.training_problem = cp.Problem(cp.Minimize(LS_lower + sparsity_penalty + sum(group_lasso_peanlty)))

    def solve_training(self, lam):
        self.lam.value = lam
        self.training_problem.solve(solver=cp.ECOS)
        return self.x.value

    def p_value(self):
        return np.array([float(self.penalty[i].value) for i in range(len(self.penalty))])

class Training_model:
    def __init__(self, data, settings) -> None:  
        n, p, M = settings.num_train, settings.num_features, settings.num_experiment_groups
        cal_group_sizes = [p//M] * M
        group_ind = np.concatenate( [[0], np.cumsum(cal_group_sizes)] )

        self.x = cp.Variable(p)
        self.r = cp.Parameter(M+1, nonneg=True)
        LS_lower = .5/n * cp.sum_squares( data.y_train - data.X_train @ self.x )
        group_lasso_constraint = [cp.pnorm(self.x[group_ind[i]:group_ind[i+1]], 2) <= self.r[i] for i in range(M)]
        sparsity_constraint = cp.pnorm(self.x, 1) <= self.r[M]
        self.training_problem = cp.Problem(cp.Minimize(LS_lower), group_lasso_constraint + [sparsity_constraint])

    def solve_training(self, r):
        self.r.value = r
        try:
            self.training_problem.solve(solver=cp.ECOS)
        except:
            self.training_problem.solve(solver=cp.SCS)
        return self.x.value

def ls_err(X, y, w):
    return np.sum(np.square(X @ w - y))/len(y)

def GS_RUN(data, lst, x_grid):
    lower_solver = Training_model(data, lst)
    monitor = Monitor() 
    T = 0 

    r = np.zeros(lst.num_experiment_groups + 1)

    for rs in x_grid:
        t0 = time.time()
        r[:-1] = rs[0] * np.ones(lst.num_experiment_groups)
        r[-1] = rs[1] 
        w = lower_solver.solve_training(r)
        T += time.time() - t0 
        monitor.append({"time": T, "Tr": .5 * ls_err(data.X_train, data.y_train, w), "Val": .5 * ls_err(data.X_validate, data.y_validate, w), "Test": .5 * ls_err(data.X_test, data.y_test, w)})
    return monitor.to_df()

def RS_RUN(data, lst, x_generator, nIter):
    lower_solver = Training_model(data, lst)
    monitor = Monitor() 
    T = 0 

    for _ in range(nIter):
        r = x_generator()
        t0 = time.time()
        w = lower_solver.solve_training(r)
        T += time.time() - t0 
        monitor.append({"time": T, "Tr": .5 * ls_err(data.X_train, data.y_train, w), "Val": .5 * ls_err(data.X_validate, data.y_validate, w), "Test": .5 * ls_err(data.X_test, data.y_test, w)})
    return monitor.to_df()

def TPE_RUN(data, lst, x_space, nIter):
    lower_solver = Training_model(data, lst)
    monitor = Monitor() 
    T = 0 
    
    def Bayesian_obj(para):
        nonlocal monitor, T 
        t0 = time.time()
        r = np.zeros(lst.num_experiment_groups + 1)
        for i in range(lst.num_experiment_groups + 1):
            r[i] = para[i]
        w = lower_solver.solve_training(r)
        T += time.time() - t0 
        monitor.append({"time": T, "Tr": .5 * ls_err(data.X_train, data.y_train, w), "Val": .5 * ls_err(data.X_validate, data.y_validate, w), "Test": .5 * ls_err(data.X_test, data.y_test, w)})
        return monitor.Test[-1]

    Best = fmin(Bayesian_obj, x_space, algo=tpe.suggest, max_evals=nIter)
    return monitor.to_df()


#%%
def run(k, seed, nTr, nVal, nTest = 300, nFeatures = 150, nGroup = 15, root = "results/"):
    np.random.seed(seed)
    lst = SGL_Setting(num_train=nTr, num_validate=nVal, num_test=nTest, num_features=nFeatures, num_experiment_groups=nGroup)
    data = load_data(root, nTr, nVal, nTest, nFeatures, nGroup, k)
    data_info = Data_with_Info(data, lst, k)

    group_size = [nFeatures//nGroup] * nGroup
    group_ind = np.concatenate([[0], np.cumsum(group_size)])

    lam1s = np.linspace(1, 10, 20)
    lam2s = np.linspace(1, 100, 20)
    filename = root + f"gLasso_{nTr}_{nVal}_{nTest}_{nFeatures}_{nGroup}_GS_{seed}.csv"
    if os.path.exists(filename):
        df_gs = pd.read_csv(filename)
    else:
        df_gs = GS_RUN(data, lst, product(lam1s, lam2s))
        df_gs.to_csv(filename)
    df = df_gs
    ind = df["Val"].idxmin()
    print(f"{'Grid':8s} Tr = {df['Tr'].iloc[ind]:.2f}, Val = {df['Val'].iloc[ind]:.2f}, Test = {df['Test'].iloc[ind]:.2f}, Time = {df['time'].iloc[-1]:.1f}")

    rand_lam = lambda : [10 * np.random.rand() for i in range(nGroup)] + [100 * np.random.rand()]
    rand_iter_a = 400
    filename = root + f"gLasso_{nTr}_{nVal}_{nTest}_{nFeatures}_{nGroup}_RS_{seed}.csv"
    if os.path.exists(filename):
        df_rs = pd.read_csv(filename)
    else:
        df_rs = RS_RUN(data, lst, rand_lam, rand_iter_a)
        df_rs.to_csv(root + f"gLasso_{nTr}_{nVal}_{nTest}_{nFeatures}_{nGroup}_RS_{seed}.csv")
    df = df_rs
    ind = df["Val"].idxmin()
    print(f"{'Random':8s} Tr = {df['Tr'].iloc[ind]:.2f}, Val = {df['Val'].iloc[ind]:.2f}, Test = {df['Test'].iloc[ind]:.2f}, Time = {df['time'].iloc[-1]:.1f}")

    tpe_space_a = [hp.uniform("lam" + str(i), 0, 10) for i in range(nGroup)] + [hp.uniform("lam" + str(nGroup), 0, 100)] 
    tpe_iter_a = 400
    filename = root + f"gLasso_{nTr}_{nVal}_{nTest}_{nFeatures}_{nGroup}_TPE_{seed}.csv"
    if os.path.exists(filename):
        df_tpe = pd.read_csv(filename)
    else:
        df_tpe = TPE_RUN(data, lst, tpe_space_a, tpe_iter_a)
        df_tpe.to_csv(root + f"gLasso_{nTr}_{nVal}_{nTest}_{nFeatures}_{nGroup}_TPE_{seed}.csv")
    df = df_tpe
    ind = df["Val"].idxmin()
    print(f"{'TPE':8s} Tr = {df['Tr'].iloc[ind]:.2f}, Val = {df['Val'].iloc[ind]:.2f}, Test = {df['Test'].iloc[ind]:.2f}, Time = {df['time'].iloc[-1]:.1f}")

    filename = root + f"gLasso_{nTr}_{nVal}_{nTest}_{nFeatures}_{nGroup}_HC_{seed}.csv"
    if os.path.exists(filename):
        df_hc = pd.read_csv(filename)
    else:
        HC_Setting = {
            "num_iters": 100,
            "initial_guess": 1e-1*np.ones(lst.num_experiment_groups + 1),
            "step_size_min": 0, 
            "decr_enough_threshold": 0
        }
        df_hc = Imlicit_Differntiation(data_info, HC_Setting)
        df_hc.to_csv(filename)
    print(f"{'IGJO':8s} Tr = {df_hc['train_error'].iloc[-1]:.2f}, Val = {df_hc['validation_error'].iloc[-1]:.2f}, Test = {df_hc['test_error'].iloc[-1]:.2f}, Time = {df_hc['time'].iloc[-1]:.1f}")

    filename = root + f"gLasso_{nTr}_{nVal}_{nTest}_{nFeatures}_{nGroup}_DC_{seed}.csv"
    if os.path.exists(filename):
        df_dc = pd.read_csv(filename)
    else:
        sgl_model = Training_model0(data, lst)
        sgl_model.solve_training(0.1*np.ones(nGroup+1))
        dc_initial = sgl_model.p_value()

        DC_Setting = {
            "TOL": 0,
            "initial_guess": dc_initial,
            "epsilon": 1e-2, 
            "beta_0": 1,
            "rho": .1,
            "MAX_ITERATION": 50,
            "c": .01,
            "delta": 5
        }
        df_dc = iP_DCA(data_info, DC_Setting)
        df_dc.to_csv(root + f"gLasso_{nTr}_{nVal}_{nTest}_{nFeatures}_{nGroup}_DC_{seed}.csv")
    print(f"{'DC':8s} Tr = {df_dc['train_error'].iloc[-1]:.2f}, Val = {df_dc['validation_error'].iloc[-1]:.2f}, Test = {df_dc['test_error'].iloc[-1]:.2f}, Tr = {df_dc['lower_train_error'].iloc[-1]:.2f}, Val = {df_dc['lower_validation_error'].iloc[-1]:.2f}, Test = {df_dc['lower_test_error'].iloc[-1]:.2f}, Time = {df_dc['time'].iloc[-1]:.1f}")

    return None



# %%
def main():
    root = "./results/SGL/"
    for k in range(20):
        print("-"*10 + "{:2d} experiments".format(k) + "-"*10)
        run(k, k, nTr = 300, nVal = 300, nTest = 300, nFeatures = 90, nGroup = 30, root = root)
        run(k, k, nTr = 300, nVal = 300, nTest = 300, nFeatures = 150, nGroup = 30, root = root)
        try:
            print("Convex Case")
            run(k, k, nTr = 100, nVal = 100, nTest = 300, nFeatures = 150, nGroup = 30, root = root)
            GAFFA(k, k, nTr = 100, nVal = 100, nTest = 300, nFeatures = 150, nGroup = 30, nIter = 30000, root = root)
            print("Strongly Convex Case")
            run(k, k, nTr = 300, nVal = 300, nTest = 300, nFeatures = 150, nGroup = 30, root = root)
            GAFFA(k, k, nTr = 300, nVal = 300, nTest = 300, nFeatures = 150, nGroup = 30, nIter = 30000, root = root)
        except:
            continue

if __name__ == "__main__":
    main()


# %%
