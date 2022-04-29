import numpy as np
from OperationsLib import Operations as op
class LinearReg():
    def __init__(self,file,test_pct=1.0):
        super().__init__()
        self.dataset = np.genfromtxt(file, delimiter=',', skip_header=0)
        if test_pct < 1.0:
            np.random.shuffle(self.dataset)
        self.x_test = (self.dataset[0:int((self.dataset.shape[0])*test_pct), :])[:, 0:int(self.dataset.shape[1]-1)]
        self.x_training = (self.dataset[int((self.dataset.shape[0])*test_pct):, :])[:, 0:int(self.dataset.shape[1]-1)] 
        self.y_test = (self.dataset[0:int((self.dataset.shape[0])*test_pct), :])[:, [-1]] 
        self.y_training = (self.dataset[int((self.dataset.shape[0])*test_pct):, :])[:, [-1]] 
                
    
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



    


#falta de_stddize, Graphs.py, checar se o gd functiona com std data
# e a 2 questao. 
    