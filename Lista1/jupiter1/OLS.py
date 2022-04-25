import numpy as np
import matplotlib.pyplot as plt
peixe_dataset = np.genfromtxt('peixe.txt', delimiter=',', skip_header=0)


px = peixe_dataset[:,[0,1]]# X matrix of input data
voo = np.ones((px.shape[0],1))#one column matrix
x = np.concatenate((voo,px),axis = 1)
y = peixe_dataset[:,[2]]
w = (np.linalg.inv(x.T @ x) @ x.T) @ y
yp = x @ w
print(x)
print(yp)
print(y)

#plt.plot(yp,'o')
#plt.plot(y,'x')
#plt.show()
rmse = np.sqrt(np.mean((y-yp)**2))
print(rmse)