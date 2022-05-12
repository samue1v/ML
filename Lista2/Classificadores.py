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
            pc0 = ADG.ADGprob(x_test_std,m_cov0,c0_mi)
            pc1 = ADG.ADGprob(x_test_std,m_cov1,c1_mi)            
            res = []
            for j in range(len(pc0)):
                if pc0[j]>pc1[j]:
                    res.append(0)
                else:
                    res.append(1)
            ac = []
            #print(y_train.T)
            for k in range(len(res)):
                ac.append(res[k]==int(y_test.T[0][k]))
            print(ac.count(1)/len(ac))


            
    @classmethod
    def ADGprob(cls,indep,m_cov,mi):
        prob_vec = []
        try:
            inv = np.linalg.inv(m_cov)
        except:
            m_cov += np.identity(m_cov.shape[0])*10e-20
            inv = np.linalg.inv(m_cov)

        for i in range(indep.shape[0]):
            prob_vec.append(-0.5*np.log(np.linalg.det(m_cov))-0.5*np.array([indep[i]-mi])@inv@(np.array([indep[i]-mi]).T) + np.log(0.5))

        return prob_vec


            
    
def main():
    #adg = ADG("teste2.csv",2)
    adg = ADG("breastcancer.csv",10)
    adg.ADGstep()

if __name__ == "__main__":
    main()

        

