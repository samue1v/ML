import numpy as np
import matplotlib.pyplot as plt
dataset = np.genfromtxt('artificial1d.csv', delimiter=',', skip_header=0)

px = dataset[:,[0]]# X matrix of input data
one_vecx = np.ones((px.shape[0],1))#one column matrix sizeof x
x = np.concatenate((one_vecx,px),axis = 1)
y = dataset[:,[1]]

#Adjusting x data
inv_stddev_vecx = np.divide(np.ones((px.shape[1],px.shape[1])), (px.std(axis=0))) * (np.identity(px.shape[1]))#vector of inverse std.deviations
mean_vec = (px.mean(axis=0))*(np.ones((px.shape[0],px.shape[1])))#vector with means
px_std = (px-mean_vec)@(inv_stddev_vecx)#px standardized
x_std = np.concatenate((one_vecx,px_std),axis = 1)#x standardized with 1 column

#Adjusting y data
std_dev_vecy = (y.std(axis=0))
inv_stddev_vecy = np.divide(np.ones((y.shape[1],y.shape[1])), (y.std(axis=0))) * (np.identity(y.shape[1]))
mean_vecy = (y.mean(axis=0))*(np.ones((y.shape[0],y.shape[1])))
y_std = (y-mean_vecy)@(inv_stddev_vecy)

w=np.zeros((1,x_std.shape[1]))#parameters initial values
alpha = 0.01
t=0
it = []
mseLst= []

while t<1000:
    t+=1
    it.append(t) 
    yp = np.sum((x_std * w).T, axis=0)
    yp = np.array([yp]).T #y_hat
    e_std = y_std-yp#error
    m_err = e_std * x_std #ei x xi
    k = m_err.mean(axis=0)*alpha #Gradient
    w = w + k
    ypo = (std_dev_vecy*yp) + mean_vecy
    e = (y-ypo) 
    mse = (np.mean(e**2))
    mseLst.append(mse)


print("parameters: ",end='')
print(w)
print("mse: ",end='')
print(mse)

plt.subplot(1,2,1)
plt.ylabel("MSE")
plt.xlabel("Iterations")
plt.plot(it,mseLst)


plt.subplot(1,2,2)
plt.ylabel("Y")
plt.xlabel("X")
plt.scatter(x[:,[1]],y)
plt.plot(x[:,[1]],ypo)

plt.gcf().canvas.set_window_title('GD')

plt.show()
    

