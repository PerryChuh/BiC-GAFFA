#%%
import cvxpy as cp
import numpy as np
import torch
from torch import nn 
from torch.autograd import grad 
from libsvmdata import fetch_libsvm
import pandas as pd 

import matplotlib.pyplot as plt
import time

# %%
def run(seed, dataset="diabetes_scale"):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    p = 0.3
    nIter = 5000 # 500
    gam1 = 10.
    gam2 = 0.01 
    # etak = 0.005
    etak = 0.01
    alphak = 0.001
    betak = 0.001
    ck0 = 10
    R = 10

    # load data
    X, y = fetch_libsvm(dataset)
    X, y = np.float32(X), np.float32(y)
    nFeature = X.shape[1]

    order = np.arange(len(y))
    np.random.seed(seed)
    np.random.shuffle(order)

    z_train, y_train = X[order[:500]], y[order[:500]]
    z_val, y_val = X[order[500:650]], y[order[500:650]]
    z_test, y_test = X[order[650:]], y[order[650:]]

    z_train, y_train = torch.tensor(z_train).to(device), torch.tensor(y_train).to(device)
    z_val, y_val = torch.tensor(z_val).to(device), torch.tensor(y_val).to(device)
    z_test, y_test = torch.tensor(z_test).to(device), torch.tensor(y_test).to(device)

    nTr, nVal, nTest = z_train.shape[0], z_val.shape[0], z_test.shape[0]

    # define the problem
    # x: c
    # y: w, b, xi 
    # z: lam1
    # theta: y' 
    # lam: lam0
    f = lambda w, xi, c : .5 * torch.sum(torch.square(w)) + .5 * (torch.square(xi) @ torch.exp(c))
    g = lambda w, b, xi: 1 - xi - y_train * (z_train @ w + b)
    relu0 = nn.ReLU()
    relu1 = nn.LeakyReLU()
    def F(w, b):
        x = y_val * (z_val @ w + b) / torch.linalg.norm(w)
        return torch.sum(relu1(2 * torch.sigmoid(-5.0 * x) - 1.0))

    def Ftest(w, b):
        x = y_test * (z_test @ w + b) / torch.linalg.norm(w)
        return torch.sum(relu1(2 * torch.sigmoid(-5.0 * x) - 1.0)).cpu().detach().numpy()

    def accVal(w, b):
        tmp = y_val * (z_val @ w + b)
        return sum(tmp>0) / nVal * 100

    def accTest(w, b):
        tmp = y_test * (z_test @ w + b)
        return sum(tmp>0) / nTest * 100

    # Locate Variable
    w0 = torch.ones(nFeature, requires_grad=True, device=device)
    b0 = torch.zeros(1, requires_grad=True, device=device)
    xi0 = torch.ones(nTr, requires_grad=True, device=device)
    w1 = torch.ones(nFeature, requires_grad=True, device=device)
    b1 = torch.zeros(1, requires_grad=True, device=device)
    xi1 = torch.ones(nTr, requires_grad=True, device=device)
    lam0 = torch.ones(nTr, device=device)
    lam1 = torch.ones(nTr, device=device)
    c0 = -6. + 1. * torch.rand(nTr, device=device, requires_grad=True)

    # Set the Initial Value
    svm_w = cp.Variable(nFeature)
    svm_b = cp.Variable()
    svm_xi = cp.Variable(nTr)
    loss = cp.sum_squares(svm_w) + cp.exp(c0.cpu().detach().numpy()) @ cp.square(svm_xi)
    constraints = [1 - svm_xi - cp.multiply(y_train.cpu().numpy(), (z_train.cpu().numpy() @ svm_w + svm_b)) <= 0]
    prob = cp.Problem(cp.Minimize(loss), constraints)

    algorithm_start_time = time.time()
    # prob.solve(solver='ECOS')
    prob.solve(solver='ECOS', abstol=2e-3,reltol=2e-3,max_iters=1000000000, warm_start=True)  

    w0.data = torch.tensor(svm_w.value, device = device, dtype=torch.float32)
    b0.data = torch.tensor(svm_b.value, device = device, dtype=torch.float32)
    xi0.data = torch.tensor(svm_xi.value, device = device, dtype=torch.float32)
    w1.data = torch.tensor(svm_w.value, device = device, dtype=torch.float32)
    b1.data = torch.tensor(svm_b.value, device = device, dtype=torch.float32)
    xi1.data = torch.tensor(svm_xi.value, device = device, dtype=torch.float32)
    lam0.data = torch.tensor(constraints[0].dual_value, device = device, dtype=torch.float32)
    lam1.data = torch.tensor(constraints[0].dual_value, device = device, dtype=torch.float32)

    ValLoss, ValAcc = [F(w0, b0).cpu().detach().numpy()/15.0], [accVal(w0, b0).cpu().detach().numpy()]
    TestLoss, TestAcc = [Ftest(w0, b0) / 11.8], [accTest(w0, b0).cpu().detach().numpy()]
    time_computation = [time.time() - algorithm_start_time]

    # print(f"{0:4d}-th iter: Val Loss = {ValLoss[-1]:.4f} Test Loss = {TestLoss[-1]: .4f} ValAcc = {ValAcc[-1]:.2f} Test Acc = {TestAcc[-1]:.2f}")

    for k in range(nIter):
        ck = ck0 * (k + 1)**p
        f1k = f(w1, xi1, c0)
        g1k = lam0 @ g(w1, b1, xi1)

        dw1 = grad(f1k, w1, retain_graph=True)[0] + grad(g1k, w1, retain_graph=True)[0] + 1 / gam1 * (w1 - w0)
        db1 = grad(g1k, b1, retain_graph=True)[0] + 1 / gam1 * (b1 - b0)
        dxi1 = grad(f1k, xi1)[0] + lam1 @ grad(g1k, xi1)[0] + 1 / gam1 * (xi1 - xi0)

        w1p = w1 - etak * dw1
        b1p = b1 - etak * db1
        xi1p = xi1 - etak * dxi1

        g0k = g(w0, b0, xi0)
        lam0 = relu0(lam1 + gam2 * g0k)

        F0k = F(w0, b0)
        f0k = f(w0, xi0, c0)
        f1k = f(w1p, xi1p, c0)
        g0k = lam0 @ g(w0, b0, xi0)

        dc0 = grad(f0k, c0, retain_graph=True)[0] - grad(f1k, c0, retain_graph=True)[0]
        c0p = c0 - alphak * dc0

        dw0 = 1/ck * grad(F0k, w0, retain_graph=True)[0] + grad(f0k, w0, retain_graph=True)[0] + grad(g0k, w0, retain_graph=True)[0] - 1 / gam1 * (w0 - w1)
        db0 = 1/ck * grad(F0k, b0, retain_graph=True)[0] + grad(g0k, b0, retain_graph=True)[0] - 1 / gam1 * (b0 - b1)
        dxi0 = grad(f0k, xi0, retain_graph=True)[0] + lam0 @ grad(g0k, xi0)[0] - 1 / gam1 * (xi0 - xi1)

        dlam1 = -(lam1 - lam0) / gam2 - g(w1, b1, xi1)
        with torch.no_grad():
            w0 = w0 - alphak * dw0
            b0 = b0 - alphak * db0
            xi0 = xi0 - alphak * dxi0
            lam1 = torch.minimum(torch.maximum(lam1 - betak * dlam1, torch.zeros(nTr)), R*torch.ones(nTr))

        w1, b1, xi1, c0 = w1p.detach().clone().requires_grad_(True), b1p.detach().clone().requires_grad_(True), xi1p.detach().clone().requires_grad_(True), c0p.detach().clone().requires_grad_(True)
        w0.requires_grad_(True)
        b0.requires_grad_(True)
        xi0.requires_grad_(True)


        ValLoss.append(F0k.cpu().detach().numpy() / 15.0)
        ValAcc.append(accVal(w0, b0).cpu().detach().numpy())
        TestLoss.append(Ftest(w0, b0) / 11.8)
        TestAcc.append(accTest(w0, b0).cpu().detach().numpy())
        time_computation.append(time.time() - algorithm_start_time)

    return ValLoss, TestLoss, ValAcc, TestAcc, time_computation

# %%
if __name__ == "__main__":
    val_loss_array=[]
    test_loss_array=[]
    val_acc_array=[]
    test_acc_array=[]
    time_array = []
    dataset="diabetes_scale"
    for seed in range(20):
        val_loss_list,test_loss_list,val_acc_list,test_acc_list,time_computation=run(seed, dataset)
        result = pd.DataFrame({"val_loss": val_loss_list, "test_loss": test_loss_list, "val_acc": val_acc_list, "test_acc": test_acc_list, "time": time_computation})
        result.to_csv(f"./results/SVM_HO_{dataset}_seed={seed}.csv", mode='w', header=True, index=False)
        val_loss_array.append(np.array(val_loss_list))
        test_loss_array.append(np.array(test_loss_list))
        val_acc_array.append(np.array(val_acc_list))
        test_acc_array.append(np.array(test_acc_list))
        time_array.append(np.array(time_computation))
    
    val_loss_array=np.array(val_loss_array)
    test_loss_array=np.array(test_loss_array)
    val_acc_array=np.array(val_acc_array)
    test_acc_array=np.array(test_acc_array)
    time_array=np.array(time_array)
    time_computation = np.mean(time_array, axis=0)

    val_loss_mean=np.sum(val_loss_array,axis=0)/val_loss_array.shape[0]
    val_loss_sd=np.sqrt(np.var(val_loss_array,axis=0))/2.0
    test_loss_mean=np.sum(test_loss_array,axis=0)/test_loss_array.shape[0]
    test_loss_sd=np.sqrt(np.var(test_loss_array,axis=0))/2.0

    val_acc_mean=np.sum(val_acc_array,axis=0)/val_acc_array.shape[0]
    val_acc_sd=np.sqrt(np.var(val_acc_array,axis=0))/2.0
    test_acc_mean=np.sum(test_acc_array,axis=0)/test_acc_array.shape[0]
    test_acc_sd=np.sqrt(np.var(test_acc_array,axis=0))/2.0

    plt.rcParams.update({'font.size': 18})
    plt.rcParams['font.sans-serif']=['Arial']
    plt.rcParams['axes.unicode_minus']=False
    axis=time_computation
    plt.figure(figsize=(8,6))
    #plt.grid(linestyle = "--") 
    ax = plt.gca()
    plt.plot(axis,val_loss_mean,'-',label="Training loss")
    ax.fill_between(axis,val_loss_mean-val_loss_sd,val_loss_mean+val_loss_sd,alpha=0.2)
    plt.plot(axis,test_loss_mean,'--',label="Test loss")
    ax.fill_between(axis,test_loss_mean-test_loss_sd,test_loss_mean+test_loss_sd,alpha=0.2)
    #plt.xticks(np.arange(0,iterations,40))
    plt.title('Liner SVM')
    plt.xlabel('Running time /s')
    #plt.legend(loc=4)
    plt.ylabel("Loss")
    #plt.xlim(-0.5,3.5)
    #plt.ylim(0.5,1.0)
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    #plt.setp(ltext, fontsize=18,fontweight='bold') 
    # plt.savefig('ho_svm_1.pdf') 
    plt.show()

    axis=time_computation
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    plt.plot(axis,val_acc_mean,'-',label="Training accuracy")
    ax.fill_between(axis,val_acc_mean-val_acc_sd,val_acc_mean+val_acc_sd,alpha=0.2)
    plt.plot(axis,test_acc_mean,'--',label="Test accuracy")
    ax.fill_between(axis,test_acc_mean-test_acc_sd,test_acc_mean+test_acc_sd,alpha=0.2) 
    #plt.xticks(np.arange(0,iterations,40))
    plt.title('Liner SVM')
    plt.xlabel('Running time /s')
    plt.ylabel("Accuracy (%)")
    plt.ylim(64, 80)
    #plt.legend(loc=4)
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    #plt.setp(ltext, fontsize=18,fontweight='bold') 
    # plt.savefig('ho_svm_2.pdf') 
    plt.show()

    print(f"Val Acc: {val_acc_mean[-1]:.2f} pm {val_acc_sd[-1]:.2f} Test Acc: {test_acc_mean[-1]:.2f} pm {test_acc_sd[-1]:.2f}")
# %%
