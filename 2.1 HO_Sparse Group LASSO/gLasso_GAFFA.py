#%%
import numpy as np 
import cvxpy as cp
import pandas as pd
import time 
import matplotlib.pyplot as plt
import os

#%%
class Data():
    def __init__(self):
        self.X_train = 0
        self.X_validate = 0
        self.X_test = 0
        self.y_train = 0
        self.y_validate = 0
        self.y_test = 0

class SGL_Setting:
    def __init__(self, num_train=50, num_validate=50, num_test=300, num_features=150, num_experiment_groups=15):
        self.num_train = num_train
        self.num_validate = num_validate
        self.num_test = num_test
        self.num_features = num_features
        self.num_experiment_groups = num_experiment_groups
        self.num_true_groups = 5

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
        self.training_problem.solve(cp.ECOS)
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
        self.constraint = group_lasso_constraint + [sparsity_constraint]
        self.training_problem = cp.Problem(cp.Minimize(LS_lower), self.constraint)

    def solve_training(self, r):
        self.r.value = r
        self.training_problem.solve(solver=cp.ECOS)
        return self.x.value

    def solve_training2(self, r):
        r1 = r
        r1[:-1] = np.sqrt(r1[:-1])
        self.r.value = r1
        self.training_problem.solve(solver=cp.ECOS)
        return self.x.value

def ls_err(X, y, w):
    return np.sum(np.square(X @ w - y))/len(y)

#%%
def run(k, seed, nTr, nVal, nTest = 300, nFeatures = 150, nGroup = 15, nIter = None, root = "D:/data/results24/lasso/"):
    # np.random.seed(seed)
    nIter = 30000 if nIter is None else nIter
    p = 0.3
    gam1, gam2 = 10, 1.
    ck0, lamU = 10, .5
    etak, alphak, betak = 1e-1, 1e-2, 1e-2

    filename = root + f"gLasso_{nTr}_{nVal}_{nTest}_{nFeatures}_{nGroup}_SPBA_{p}_{seed}.csv"
    lst = SGL_Setting(num_train=nTr, num_validate=nVal, num_test=nTest, num_features=nFeatures, num_experiment_groups=nGroup)
    data = load_data(root, nTr, nVal, nTest, nFeatures, nGroup, k)

    group_size = [nFeatures//nGroup] * nGroup
    group_ind = np.concatenate([[0], np.cumsum(group_size)])

    F = lambda w: .5 * ls_err(data.X_validate, data.y_validate, w) 
    dwF = lambda w: data.X_validate.T @ (data.X_validate @ w - data.y_validate) / len(data.y_validate)

    f = lambda w: .5 * ls_err(data.X_train, data.y_train, w)
    dwf = lambda w: data.X_train.T @ (data.X_train @ w - data.y_train) / len(data.y_train)

    g = lambda r, w: np.concatenate([
    [np.sum(np.square(w[group_ind[i]:group_ind[i+1]])) - r[i] for i in range(nGroup)], 
    [(np.linalg.norm(w, 1) - r[nGroup])]
    ])
    dwlamg = lambda r, w, lam: np.concatenate([2 * w[group_ind[i]:group_ind[i+1]] * lam[i] for i in range(nGroup)]) + np.sign(w) * lam[nGroup]
    drlamg = lambda w, r, lam: - lam

    lower_solver = Training_model(data, lst)
    
    lower0  = Training_model0(data, lst)
    w0 = lower0.solve_training(.1*np.ones(nGroup+1))
    r0 = lower0.p_value()
    r0[:-1] = np.square(r0[:-1])
    w1 = np.copy(w0)
    lam0 = 0.1 * np.ones(nGroup + 1) 
    lam1 = 0.1 * np.ones(nGroup + 1) 

    monitor = Monitor() 

    T = 0 
    time0 = time.time() 

    r1 = r0
    r1[:-1] = [r**2 for r in r1[:-1]]
    w_hat = lower_solver.solve_training2(r0)
    T += time.time() - time0
    monitor.append({"k": 0, "time": T, 
                    "Val": F(w0), "Tr": f(w0), 
                    "Test": .5 * ls_err(data.X_test, data.y_test, w0), 
                    "dy": np.linalg.norm(w0 - w_hat),
                    "g": np.linalg.norm(np.maximum(g(r0, w0), 0)), 
                    })

    for k in range(nIter):
        time0 = time.time()
        # ck = np.min([ck0 * (k + 1) ** p, 50])
        ck = ck0 * (k + 1) ** p

        for _ in range(1):
            dw1 = dwf(w0) + dwlamg(r0, w0, lam0) + 1 / gam1 * (w1 - w0)
            w1p = w1 - etak * dw1

            lam0 = lam1 + gam2 * g(r0, w0)
            lam0 = np.maximum(0, lam0)
        
        dr = drlamg(r0, w0, lam0) - drlamg(r0, w1p, lam1)
        r0 = r0 - alphak * dr

        dw0 = 1 / ck * dwF(w0) + dwf(w0) + dwlamg(r0, w0, lam0) - (w0 - w1p) / gam1
        w0 = w0 - alphak * dw0

        dlam1 = - (lam1 - lam0) / gam2 - g(r0, w1p)
        lam1 = lam1 - betak * dlam1
        lam1 = np.minimum(np.maximum(0, lam1), lamU)

        w1 = w1p

        T += time.time() - time0 
        # w_hat = lower_solver.solve_training2(r0)
        monitor.append({"k": k+1, "time": T,
                        "Val": F(w0), "Tr": f(w0), 
                        "Test": .5 * ls_err(data.X_test, data.y_test, w0),
                        "dy": np.linalg.norm(w0 - w_hat),
                        "g": np.linalg.norm(np.maximum(g(r0, w0), 0)), 
                        })

    w_hat = lower_solver.solve_training2(r0)
    monitor.append({"k": k+1, "time": T,
                    "Val": F(w0), "Tr": f(w0), 
                    "Test": .5 * ls_err(data.X_test, data.y_test, w0),
                    "dy": np.linalg.norm(w0 - w_hat),
                    "g": np.linalg.norm(np.maximum(g(r0, w0), 0)), 
                    })

    print(f'{"SPBA":8s} Tr = {monitor.Tr[-1]:.2f}, Val = {monitor.Val[-1]:.2f}, Test = {monitor.Test[-1]:.2f}')

    df = monitor.to_df()
    df.to_csv(filename)

# %%
def main():
    root = "./results/SGL/"
    for k in range(20):
        print("-"*10 + "{:2d} experiments".format(k) + "-"*10)
        run(k, k, nTr = 100, nVal = 100, nTest = 300, nFeatures = 150, nGroup = 30, nIter = 50000, root = root)
        run(k, k, nTr = 300, nVal = 300, nTest = 300, nFeatures = 150, nGroup = 30, nIter = 50000, root = root)

if __name__ == "__main__":
    main()