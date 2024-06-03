import numpy as np
import pandas as pd

class Monitor():
    def __init__(self):
        self.time = []
        self.train_error = []
        self.validation_error = []
        self.test_error = []
    
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

class Monitor_DC(Monitor):
    def __init__(self):
        super().__init__()
        self.step_err = []
        self.penalty = []
        self.beta = []
        self.lower_train_error = []
        self.lower_validation_error = []
        self.lower_test_error = []
