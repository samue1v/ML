import numpy as np
from OperationsLib import Operations as op

class LogisticReg:
    def __init__(self,file,n_folds):
        self.dataset = np.genfromtxt(file, delimiter=',', skip_header=0)
        self.n_folds = n_folds
        self.folds = op.Kfold(self.dataset,self.n_folds)

    

class GDLog(LogisticReg):
    def __init__(self, file, n_folds,pace,it_num):
        super().__init__(file, n_folds)
        self.pace = pace
        self.it_num = it_num
        self.results_vec = []
    
    def LogStep(self):
        for i in range(self.n_folds):
            t=0
            test,train = op.get_folded_data(self.folds,i)
            x_train,y_train = op.slice_data(train)
            x_test,y_test = op.slice_data(test)
            y_n = y_train #op.stddize(y_test)
            x_n = op.stddize(x_train) #op.stddize(x_test)#nxd
            x_coTrain = op.concat_one_column(x_n)#nxd+1
            x_coTest = op.concat_one_column(op.stddize(x_test))#nxd+1
            self.w = np.zeros((x_coTrain.shape[1],1))#d+1x1
            #print(x_n.shape)
            while t< self.it_num:
                t+=1
                #print(x_train.shape)
                y_hat = x_coTrain @ self.w      #np.array([np.sum(self.w * x_co,axis=1)]).T
                err = y_n - op.sigmoid(y_hat)
                self.w = self.w + self.pace*(np.array([np.mean(err*x_coTrain,axis=0)])).T
                #y_hot = op.de_stddize(self.y_test,y_hat)
                #e = self.y_test - y_hot
                #mse = np.mean(e**2)
                #self.mse_list.append(mse)
            y_hatTest = x_coTest @ self.w
            for _ in y_hatTest:
                for i in range(y_hatTest.shape[0]):
                    if y_hatTest[i]>=0.5:
                        y_hatTest[i]=1
                    else:
                        y_hatTest[i] = 0
                    self.results_vec.append(int(y_hatTest[i])==int(y_test[i]))
            
            c = self.results_vec.count(1)/len(self.results_vec)
            print(c)
        




def main():
    l = GDLog("breastcancer.csv",10,0.01,100)
    l.LogStep()
    #print(l.results_vec)


if __name__ == "__main__":
    main()

