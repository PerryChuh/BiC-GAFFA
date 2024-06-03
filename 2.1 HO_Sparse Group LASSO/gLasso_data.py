#%%
import numpy as np 
import cvxpy as cp
import pandas as pd
import time 
import matplotlib.pyplot as plt
from hyperopt import tpe, hp, fmin
from itertools import product

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
    def __init__(self, num_train=50, num_validate=50, num_test=300, num_features=150, num_experiment_groups=15, normalized = True):
        self.num_train = num_train
        self.num_validate = num_validate
        self.num_test = num_test
        self.num_features = num_features
        self.num_experiment_groups = num_experiment_groups
        self.num_true_groups = 5
        self.normalized = normalized
        # num_true_groups is defined in data_generator

def SGL_Data_Generator(settings):
    num_train, num_validate, num_test = settings.num_train, settings.num_validate, settings.num_test
    num_features = settings.num_features
    num_samples = num_train + num_validate + num_test

    num_true_groups = settings.num_true_groups

    group_feature_sizes = [num_features//num_true_groups] * num_true_groups
    base_nonzero_coeff = np.array([1, 2, 3, 4, 5])
    num_nonzero_features = len(base_nonzero_coeff)

    X = np.random.randn(num_samples, num_features)

    beta_real = np.concatenate(
        [np.concatenate( (base_nonzero_coeff, np.zeros(group_feature_size - num_nonzero_features)) )
        for group_feature_size in group_feature_sizes] )

    y_true = X @ beta_real

    # add noise
    snr = 3
    epsilon = np.random.randn(num_samples)
    SNR_factor = snr / np.linalg.norm(y_true) * np.linalg.norm(epsilon)
    y = y_true + 1.0 / SNR_factor * epsilon

    data = Data()
    # split data
    data.X_train, data.X_validate, data.X_test = X[0:num_train], X[num_train:num_train + num_validate], X[num_train + num_validate:]
    data.y_train, data.y_validate, data.y_test = y[0:num_train], y[num_train:num_train + num_validate], y[num_train + num_validate:]

    return data

#%%
def run(k, seed, nTr, nVal, nTest = 300, nFeatures = 150, nGroup = 15, root = "results/", normalized = True):
    np.random.seed(seed)
    lst = SGL_Setting(num_train=nTr, num_validate=nVal, num_test=nTest, num_features=nFeatures, num_experiment_groups=nGroup, normalized=normalized)
    data = SGL_Data_Generator(lst)
    if normalized:
        np.save(root + f"data_scaled_nTr={nTr}_nVal={nVal}_nTest={nTest}_nFeatures={nFeatures}_nGroup={nGroup}_k={k}.npy", data.__dict__)
    else:
        np.save(root + f"data_nTr={nTr}_nVal={nVal}_nTest={nTest}_nFeatures={nFeatures}_nGroup={nGroup}_k={k}.npy", data.__dict__)


# %%
def main():
    root1 = "./results/"
    for k in range(20):
        print("-"*10 + "{:2d} experiments".format(k) + "-"*10)
        run(k, k, nTr = 300, nVal = 300, nTest = 300, nFeatures = 150, nGroup = 30, root = root1)
        run(k, k, nTr = 100, nVal = 100, nTest = 300, nFeatures = 150, nGroup = 30, root = root1)

if __name__ == "__main__":
    main()
# %%
