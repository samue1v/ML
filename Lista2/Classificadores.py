import numpy as np
from OperationsLib import Operations as op
class Classificadores():
    def __init__(self,file,n_folds):
        self.dataset = np.genfromtxt(file, delimiter=',', skip_header=0)
        self.n_folds = n_folds
        self.folds = op.Kfold(self.dataset,self.n_folds)

class ADG(Classificadores):
    def __init__(self,file,n_folds):
        super().__init__(file,n_folds)
    
    def ADGstep(self):
        for i in range(self.n_folds):
            c0=[]
            c1=[]
            test,train = op.get_folded_data(self.folds,i)
            x_train,y_train = op.slice_data(train)
            x_test,y_test = op.slice_data(test)
            x_train_std = op.stddize(x_train)
            x_test_std = op.stddize(x_test)
            for j in range(y_train):
                if y_train[j]==0:
                    c0.append(x_train[j])
                if y_train[j]==1:
                    c1.append(x_train[j])
        

