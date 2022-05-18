from OperationsLib import Operations as op
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
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
    
    def KNNstep(self):
        for i in range(self.n_folds):
            test,train = op.get_folded_data(self.folds,i)
            x_train,y_train = op.slice_data(train)
            x_test,y_test = op.slice_data(test)
            x_train_std = op.stddize(x_train)
            x_test_std = op.stddize(x_test)
            
            y_hat = KNN.calcKNN(x_train_std,x_test_std,y_train,self.k_nearest)
            y_hat = np.array(y_hat)
            y_test = np.reshape(y_test,(1,y_hat.shape[0]))[0]
            self.revoc.append(metrics.recall_score(y_test,y_hat))
            self.acuracia.append(metrics.accuracy_score(y_test,y_hat))
            self.precisao.append(metrics.precision_score(y_test,y_hat))
            self.f1.append(metrics.f1_score(y_test,y_hat))


    @classmethod
    def calcKNN(cls,x_train,x_test,y_train,k_nearest):
        res = []
        for i in range(x_test.shape[0]):
            k_near = []
            #s = np.square((x_train-x_test[i]))
            #s = np.sqrt(np.sum(s,axis=1))
            
            s = np.linalg.norm(x_train-x_test[i],axis=1)
            s=np.reshape(s,(s.shape[0],1))
            max = np.amax(s)  #maior distancia euclidiana
            for j in range(k_nearest):
                mini = np.argmin(s)#posicao da menor distancia euclidiana
                k_near.append(y_train[mini][0])
                s[mini][0] = max#substituo a atual menor distancia pela maior para não repeti-la
            res.append(op.find_most_freq(k_near))#dentre os k menores, achar aquele que mais prevalece no conjunto k_near
        return res

def main():
    knn = KNN("kc2.csv",10,5)
    knn.KNNstep()
    op.show_score((knn.acuracia),knn.algo,"Acuracia")
    op.show_score((knn.revoc),knn.algo,"Revocação")
    op.show_score((knn.precisao),knn.algo,"precisão")
    op.show_score((knn.f1),knn.algo,"F1-Score")

if __name__ == "__main__":
    main()