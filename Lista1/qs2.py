import numpy as np
import matplotlib.pyplot as plt
dataset = np.genfromtxt('california.csv', delimiter=',', skip_header=0)

np.random.shuffle(dataset)
test = (dataset[0:int((dataset.shape[0])*0.2), :])[:, 0:int(dataset.shape[1]-1)] 
testy = (dataset[0:int((dataset.shape[0])*0.2), :])[:,int(dataset.shape[1]-1):int(dataset.shape[1])] 
training = (dataset[int((dataset.shape[0])*0.2):, :])[:, 0:int(dataset.shape[1]-1)] 
trainingy = (dataset[int((dataset.shape[0])*0.2):, :])[:, int(dataset.shape[1]-1):int(dataset.shape[1])] 



print("shape of test",end='')
print(test.shape)
print("shape of y_test",end='')
print(testy.shape)

print("shape of training",end='')
print(training.shape)
print("shape of y_training",end='')
print(trainingy.shape)