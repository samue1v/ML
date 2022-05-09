import numpy as np
from OperationsLib import Operations as op
class LinearRegcls():
    def __init__(self,file,test_pct=1.0,n_folds=1):
        self.dataset = np.genfromtxt(file, delimiter=',', skip_header=0)
        self.x_training = (self.dataset[0:int((self.dataset.shape[0])*test_pct) , 0:-1])#[:, 0:int(self.dataset.shape[1]-1)]
        self.x_test = (self.dataset[int((self.dataset.shape[0])*test_pct):, 0:-1])#[:, 0:int(self.dataset.shape[1]-1)] 
        self.y_training = (self.dataset[0:int((self.dataset.shape[0])*test_pct), -1:])#[:, [-1]] 
        self.y_test = (self.dataset[int((self.dataset.shape[0])*test_pct):, -1:])#[:, [-1]] 
        self.n_folds = n_folds
        


    
    def get_xTraining(self, has_one_column : bool = False):
        if(has_one_column):
            self.x_training = op.concat_one_column(self.x_training)
        return self.x_training

    
    def get_xTraining(self, has_one_column : bool = False):
        if(has_one_column):
            self.x_training = op.concat_one_column(self.x_training)
        return self.x_training

    def get_xTest(self, has_one_column : bool = False):
        if(has_one_column):
            self.x_test = op.concat_one_column(self.x_test)
        return self.x_test

    def get_yTest(self):
        return self.y_test
    
    def get_yTraining(self):
        return self.y_training

    