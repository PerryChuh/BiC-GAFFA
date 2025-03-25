#%%
import pandas as pd

class Monitor():
    def __init__(self) -> None:
        self.k = []
        self.time = []
        self.F = []
        self.f = []
        self.g = []
        self.dx = []
        self.dy = []
    
    def append(self, data_dic):
        for attr in self.__dict__.keys():
            self.append_one(data_dic, attr)
    
    def append_one(self, data_dic, attr):
        if attr in data_dic.keys():
            self.__dict__[attr].append(data_dic[attr])
        else:
            self.__dict__[attr].append(0)

    def save_csv(self, filename):
        return pd.DataFrame(self.__dict__).to_csv(filename, mode='w', header=True, index=False)


class GAFFA_Setting():
    def __init__(self, n = 100, P = 1, p = 0.2, nIter = 30000, gam1 = 1., gam2 = 1., etak = 0.01, betak = 0.01, alphak = 0.001, R = 1, ck0 = 1, eps = 0) -> None:
        self.n = n
        self.P = P
        self.p = p
        self.nIter = nIter
        self.gam1 = gam1 
        self.gam2 = gam2 
        self.etak = etak 
        self.betak = alphak 
        self.alphak = alphak 
        self.eps = eps
        self.R = R
        self.ck0 = ck0


#%%