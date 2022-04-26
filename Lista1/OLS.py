import numpy as np
import matplotlib.pyplot as plt
dataset = np.genfromtxt('artificial1d.csv', delimiter=',', skip_header=0)


px = dataset[:,[0]]# X matrix of input data
voo = np.ones((px.shape[0],1))#one column matrix
x = np.concatenate((voo,px),axis = 1)
y = dataset[:,[1]]
w = (np.linalg.inv(x.T @ x) @ x.T) @ y
yp = x @ w

mse = (np.mean((y-yp)**2))


print("parameters: ",end='')
print(w.T)
print("mse: ",end='')
print(mse)

plt.ylabel("Y")
plt.xlabel("X")
plt.scatter(x[:,[1]],y)
plt.plot(x[:,[1]],yp)

plt.gcf().canvas.set_window_title('OLS')

plt.show()