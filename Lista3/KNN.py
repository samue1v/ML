from OperationsLib import Operations as op
import numpy as np
class N_parametrico:
    def __init__(self,file,n_folds):
        self.dataset = (np.genfromtxt(file, delimiter=',', skip_header=0))
        np.random.shuffle(self.dataset)
        self.n_folds = n_folds
        self.folds = op.Kfold(self.dataset,self.n_folds)
        self.acuracia = []
        self.revoc = []
        self.precisao = []
        self.f1 = []
    
class KNN(N_parametrico):
    def __init__(self,file,n_folds,k_nearest):
        super().__init__(file,n_folds)
        self.k_nearest = k_nearest
        self.algo = "KNN"
        self.vec_menores_dist = []
    
    def KNNstep(self):
        for i in range(self.n_folds):
            test,train = op.get_folded_data(self.folds,i)
            x_train,y_train = op.slice_data(train)
            x_test,y_test = op.slice_data(test)
            
            x_train_std = op.stddize(x_train)
            x_test_std = op.stddize(x_test)
            y_hat = KNN.calcKNN(x_train_std,x_test_std,y_train)
            print(y_hat)
            break


    @classmethod
    def calcKNN(cls,x_train,x_test,y_train):
        res = []
        for i in range(x_test.shape[0]):
            s = (x_train-x_test[i])**2
            s = np.sum(s,axis=1)
            maxi = np.argmin(s) 
            res.append(y_train[maxi][0])
        return res

def main():
    knn = KNN("kc2.csv",10,1)
    knn.KNNstep()

if __name__ == "__main__":
    main()