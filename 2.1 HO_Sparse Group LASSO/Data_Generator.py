import numpy as np
from libsvmdata import fetch_libsvm

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

def Data_Generator_Wrapper(generator, settings, data_index=0):
    data_info = Data_with_Info(generator(settings), settings)
    data_info.data_index = data_index
    return data_info 

#%%
class SGL_Setting:
    def __init__(self, num_train=90, num_validate=30, num_test=200, num_features=600, num_experiment_groups=30):
        self.num_train = num_train
        self.num_validate = num_validate
        self.num_test = num_test
        self.num_features = num_features
        self.num_experiment_groups = num_experiment_groups
        self.num_true_groups = 3
        # num_true_groups is defined in data_generator

def SGL_Data_Generator(settings):
    num_train, num_validate, num_test = settings.num_train, settings.num_validate, settings.num_test
    num_features = settings.num_features
    num_samples = num_train + num_validate + num_test

    num_true_groups = settings.num_true_groups

    group_feature_sizes = [num_features//num_true_groups] * num_true_groups
    base_nonzero_coeff = np.array([1, 2, 3, 4, 5])
    num_nonzero_features = len(base_nonzero_coeff)

    # correlation_matrix = [  [np.power(0.5, abs(i - j)) for i in range(0, num_features)] 
    #     for j in range(0, num_features)]
    # X = np.random.randn(num_samples, num_features) @ np.linalg.cholesky(correlation_matrix).T

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
