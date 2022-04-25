import numpy as np
import matplotlib.pyplot as plt
peixe_dataset = np.genfromtxt('peixe.txt', delimiter=',', skip_header=0)

px = peixe_dataset[:,[0,1]]# X matrix of input data
one_vecx = np.ones((px.shape[0],1))#one column matrix
x = np.concatenate((one_vecx,px),axis = 1)
y = peixe_dataset[:,[2]]


std_vecx = np.divide(np.ones((px.shape[1],px.shape[1])), (px.std(axis=0))) * (np.identity(px.shape[1]))#vector of inverse std.deviations
mean_vec = (px.mean(axis=0))*(np.ones((px.shape[0],px.shape[1])))#vector with means
px_std = (px-mean_vec)@(std_vecx)#px standardized
x_std = np.concatenate((one_vecx,px_std),axis = 1)

std_dev_vecy = (y.std(axis=0))
inv_stddev_vecy = np.divide(np.ones((y.shape[1],y.shape[1])), std_dev_vecy) * (np.identity(y.shape[1]))
mean_vecy = (y.mean(axis=0))*(np.ones((y.shape[0],y.shape[1])))
y_std = (y-mean_vecy)@(inv_stddev_vecy)

w=np.zeros((1,x_std.shape[1]))#parameters initial values
alpha = 0.1
t=0
it = []
rmseLst= []

while t<100:
    t+=1
    it.append(t)

    j = (x_std * w).T
    yp = np.sum(j, axis=0)
    yp = np.array([yp]).T #y_hat
    e_std = y_std-yp#error
    m_err = e_std * x_std #ei x xi
    k = m_err.mean(axis=0)*alpha #independent of w
    w = w + k
    ypo = (std_dev_vecy*yp) + mean_vecy
    e = (y-ypo) 
    rmse = np.sqrt(np.mean(e_std**2))
    rmseLst.append(rmse)
plt.ylabel("RMSE")
plt.xlabel("Iterations")
plt.plot(it,rmseLst)
plt.show()
    

