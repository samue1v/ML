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
            test,train = op.get_folded_data(self.folds,i)
            x_train,y_train = op.slice_data(train)
            x_test,y_test = op.slice_data(test)
            x_train_std = op.stddize(x_train)
            x_test_std = op.stddize(x_test)
            s_train = np.c_[x_train_std,y_train] #var independentes normalizadas concatenadas das dependes
            s_test = np.c_[x_test_std,y_test]
            r0= np.where((s_train[:,-1])==0)
            r1= np.where((s_train[:,-1])==1)
            indepc0 = s_train[r0][:,0:-1]#indep data
            indepc1 = s_train[r1][:,0:-1]
            c0_mi = np.mean(indepc0,axis=0)#media apenas as independentes para r[-1] = 0
            c1_mi = np.mean(indepc1,axis=0)
            m_cov0 = op.sum_covm(indepc0,c0_mi)
            m_cov1 = op.sum_covm(indepc1,c1_mi)
    
def main():
    adg = ADG("breastcancer.csv",10)
    adg.ADGstep()

if __name__ == "__main__":
    main()

        

