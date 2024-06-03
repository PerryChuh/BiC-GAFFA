# %% 
import cvxpy as cp
import numpy as np
import time
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn import functional as F

from libsvmdata import fetch_libsvm
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def run(seed, dataset="diabetes_scale"):
    # load data
    X, y = fetch_libsvm(dataset) # X: (768, 8), y: (768,)
    X, y = np.float32(X), np.float32(y)

    order = np.arange(len(y))
    np.random.seed(seed)
    np.random.shuffle(order)

    z_train, y_train = X[order[:500]], y[order[:500]]
    z_val, y_val = X[order[500:650]], y[order[500:650]]
    z_test, y_test = X[order[650:]], y[order[650:]]

    c_array = torch.Tensor(z_train.shape[0]).uniform_(-6.0,-5.0)
    c_array_tensor = torch.exp(c_array)

    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)
    print(c_array_tensor.shape)

    feature = X.shape[1]

    w = cp.Variable(feature)
    b = cp.Variable()
    xi = cp.Variable(y_train.shape[0])
    C = cp.Parameter(y_train.shape[0], nonneg=True)
    loss =  0.5*cp.norm(w, 2)**2 + 0.5 * (cp.scalar_product(C, cp.power(xi,2)))

    # Create two constraints.
    constraints=[]
    constraints_value=[]
    for i in range(y_train.shape[0]):
        constraints.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, z_train[i])+b) <= 0)
        constraints_value.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, z_train[i])+b) )

    # Form objective.
    obj = cp.Minimize(loss)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)

    val_loss_list=[]
    test_loss_list=[]
    val_acc_list=[]
    test_acc_list=[]
    time_computation=[]
    algorithm_start_time=time.time()


    for j in range(80):
        epsilon=0.005
        
        C.value=c_array_tensor.detach().numpy()

        begin=time.time()
        prob.solve(solver='ECOS', abstol=1e-25,reltol=1e-20,max_iters=1000000000, warm_start=True)
        end=time.time()
        print("time: ",end-begin)

        
        dual_variables= np.array([ constraints[i].dual_value for i in range(len(constraints))])
        constraints_value_1= np.array([ constraints_value[i].value for i in range(len(constraints))])
        #print("dual variables", dual_variables)
        #print("constraints_value ", constraints_value_1)
        # print("w value:", (w.value))
        # print("b value:", (b.value))
        #print("xi value:", (xi.value))



        number_right=0
        for i in range(len(y_val)):
            q=y_val[i] * (cp.scalar_product(w, z_val[i])+b)
            if q.value>0:
                number_right=number_right+1
        val_acc=number_right/len(y_val)
        # print(val_acc)

        number_right=0
        for i in range(len(y_test)):
            q=y_test[i] * (cp.scalar_product(w, z_test[i])+b)
            if q.value>0:
                number_right=number_right+1
        test_acc=number_right/len(y_test)
        # print(test_acc)

        w_tensor=torch.Tensor(np.array([w.value])).requires_grad_()
        b_tensor=torch.Tensor(np.array([b.value])).requires_grad_()

        x = torch.reshape(torch.Tensor(y_val), (torch.Tensor(y_val).shape[0],1)) * F.linear(torch.Tensor(z_val), w_tensor, b_tensor) / torch.linalg.norm(w_tensor)
        relu = nn.LeakyReLU()
        loss_upper= torch.sum(relu(2*torch.sigmoid(-5.0*x)-1.0))
        x1 = torch.reshape(torch.Tensor(y_test), (torch.Tensor(y_test).shape[0],1)) * F.linear(torch.Tensor(z_test), w_tensor, b_tensor) / torch.linalg.norm(w_tensor)
        test_loss_upper= torch.sum(relu(2*torch.sigmoid(-5.0*x1)-1.0))    

        if j == 0:
            val_loss_list.append(loss_upper.detach().numpy()/15.0)
            test_loss_list.append(test_loss_upper.detach().numpy()/11.8)
            val_acc_list.append(  val_acc * 100)
            test_acc_list.append(test_acc * 100)
            time_computation.append(0)
            

        inactive_constraint_list=[]
        for i in range(len(y_train)):
            if constraints_value_1[i]<-0.00001:
                inactive_constraint_list.append(i)
        print(len(inactive_constraint_list))

        active_constraint_list=[]
        for i in range(len(y_train)):
            if dual_variables[i]>0.00001:
                active_constraint_list.append(i)
        print(len(active_constraint_list))

        print("value:",(obj.value))

        #M = np.zeros((feature+1+y_train.shape[0]+len(active_constraint_list),feature+1+y_train.shape[0]+len(active_constraint_list)), dtype = float) 
        
        v1=np.ones((feature,))
        v2=np.zeros((1,))
        v3=c_array_tensor.detach().numpy()
        M1= np.diag(np.hstack((v1,v2,v3)))
        M2 = np.empty([0,0], dtype = float) 
        #v4= np.zeros((1, feature+1+y_train.shape[0]+len(active_constraint_list) ), dtype = float) 

        M2_list=[]
        for i in range(y_train.shape[0]):
            if i in active_constraint_list:
                M2_list.append( np.array([ np.hstack((z_train[i]* (-y_train[i]),np.array([-y_train[i]]),-np.eye(y_train.shape[0])[i])) ]) )
        M2= np.vstack(M2_list)

        M3= np.transpose(M2)
        M4 = np.zeros((len(active_constraint_list),len(active_constraint_list)))
        M = np.hstack((np.vstack((M1,M2)), np.vstack((M3,M4))))
        #print(M.shape)
        #print(np.linalg.matrix_rank(M))
        
        n1=np.zeros((feature+1, y_train.shape[0]))
        n2=np.diag(np.array(xi.value)*c_array_tensor.detach().numpy())
        n3=np.zeros((len(active_constraint_list),y_train.shape[0]))
        N=np.vstack((n1,n2,n3))
        #print(N.shape)

        d=-np.dot(np.linalg.inv(M), N) 
        
        d1=d[0:feature+1,]
        d2=d[feature+1:feature+1+y_train.shape[0],]
        d3=d[feature+1+y_train.shape[0]:feature+1+y_train.shape[0]+len(active_constraint_list),]
        #print(d1.shape)
        #print(d2.shape)
        #print(d3.shape)
        
        '''

        d_1=d[0:feature+1+y_train.shape[0],]
        MM=np.empty([0,0], dtype = float) 
        for i in range(y_train.shape[0]):
            if MM.shape[0]==0:
                MM= np.array([ np.hstack((z_train[i]* (-y_train[i]),np.array([-y_train[i]]),-np.eye(y_train.shape[0])[i])) ])
            else:
                MM= np.vstack((MM, np.array([ np.hstack((z_train[i]* (-y_train[i]),np.array([-y_train[i]]),-np.eye(y_train.shape[0])[i])) ]) ))
        d_y=np.transpose(np.dot(np.transpose(d_1),np.transpose(MM)))

        number_active=0
        number_inactive=0

        no_strictly_active=[]
        for i in range(y_train.shape[0]):
            if i in active_constraint_list:
                #print("active")
                d_lambda_norm=np.linalg.norm(d3[number_active])
                lambda_value = dual_variables[i]
                #print(constraints_value_1[i])
                number_active=number_active+1
                #print('d_lambda_norm ',d_lambda_norm)
                #print('lambda_value ',lambda_value)
                if d_lambda_norm*epsilon > lambda_value:
                    print(i)
                    no_strictly_active.append(i)
                    print("active")
                    print('d_lambda_norm ',d_lambda_norm)
                    print('lambda_value ',lambda_value)
                    #print(constraints_value_1[i])
                    #input()
                

            elif i in inactive_constraint_list:
                #print("inactive")
                d_y_norm=np.linalg.norm(d_y[i])
                #print(dual_variables[i])
                y_value=constraints_value_1[i]
                number_inactive=number_inactive+1
                #print('d_y_norm ',d_y_norm)
                #print('y_value ',y_value)
                if -d_y_norm*epsilon< y_value:
                    #print(dual_variables[i])
                    print(i)
                    no_strictly_active.append(i)
                    print("inactive")
                    print('d_y_norm ',d_y_norm)
                    print('y_value ',y_value)
                    #input()

            else:
                print("active/inactive")
                input()

        '''

        alpha=0.03
        print("val_upper_loss: ", loss_upper.detach().numpy()/15.0)
        print("test_loss_upper: ", test_loss_upper.detach().numpy()/11.8)
        
    
        loss_upper.backward()
        grads_w = w_tensor.grad.detach().numpy()[0]
        grads_b = b_tensor.grad.detach().numpy()
        grad=np.hstack((grads_w,grads_b))
        grad=np.reshape(grad,(1,grad.shape[0]))
        grad_update=np.dot(grad,d1)[0]
        c_array=c_array-alpha*grad_update
        c_array_tensor=torch.exp(c_array)
        
        val_loss_list.append(loss_upper.detach().numpy()/15.0)
        test_loss_list.append(test_loss_upper.detach().numpy()/11.8)
        val_acc_list.append(val_acc*100)
        test_acc_list.append(test_acc*100)
        time_computation.append(time.time()-algorithm_start_time)

    return val_loss_list,test_loss_list,val_acc_list,test_acc_list,time_computation

# %%
if __name__ == "__main__":
    val_loss_array=[]
    test_loss_array=[]
    val_acc_array=[]
    test_acc_array=[]
    dataset="diabetes_scale"
    for seed in range(20):
        val_loss_list,test_loss_list,val_acc_list,test_acc_list,time_computation=run(seed, dataset)
        result = pd.DataFrame({"val_loss": val_loss_list, "test_loss": test_loss_list, "val_acc": val_acc_list, "test_acc": test_acc_list, "time": time_computation})
        result.to_csv(f"./results/SVM_GAM_{dataset}_seed={seed}.csv", mode='w', header=True, index=False)
        val_loss_array.append(np.array(val_loss_list))
        test_loss_array.append(np.array(test_loss_list))
        val_acc_array.append(np.array(val_acc_list))
        test_acc_array.append(np.array(test_acc_list))
        time_computation=np.array(time_computation)

    val_loss_array=np.array(val_loss_array)
    test_loss_array=np.array(test_loss_array)
    val_acc_array=np.array(val_acc_array)
    test_acc_array=np.array(test_acc_array)

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
