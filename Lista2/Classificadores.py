import numpy as np
from OperationsLib import Operations as op
from sklearn import metrics
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
            #print(indepc0.shape)
            #break
            m_cov0 = op.sum_covm(indepc0,c0_mi)
            m_cov1 = op.sum_covm(indepc1,c1_mi)
            priori0 = np.count_nonzero(y_train == 0)/y_train.shape[0]
            priori1 = 1-priori0
            #print(y_train.shape[1])
            #print(priori0)
            #break
            #print(y_train)
            pc0 = ADG.ADGprob(x_test_std,m_cov0,c0_mi,priori0)
            pc1 = ADG.ADGprob(x_test_std,m_cov1,c1_mi,priori1)            
            res = []
            for j in range(len(pc0)):
                if pc0[j]>pc1[j]:
                    res.append(0)
                else:
                    res.append(1)
            print(f"Summary for the classifier ADG with accuracy {metrics.accuracy_score(y_test,res):.2f}")
            print(metrics.classification_report(y_test, res))
    
    @classmethod
    def ADGprob(cls,indep,m_cov,mi,priori):
        prob_vec = []
        try:
            inv = np.linalg.inv(m_cov)
        except:
            m_cov += np.identity(m_cov.shape[0])*10e-20
            inv = np.linalg.inv(m_cov)

        for i in range(indep.shape[0]):
            prob_vec.append(-0.5*np.log(np.linalg.det(m_cov))-0.5*np.array([indep[i]-mi])@inv@(np.array([indep[i]-mi]).T) + np.log(priori))

        return prob_vec

class NBG(Classificadores):
    def __init__(self,file,n_folds):
        super().__init__(file,n_folds)
    
    def NBGstep(self):
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
            var0 = op.calc_var(indepc0,c0_mi)
            var1 = op.calc_var(indepc1,c1_mi)
            priori0 = np.count_nonzero(y_train == 0)/y_train.shape[0]
            priori1 = 1-priori0
            pc0 = NBG.NBGprob(x_test_std,var0,c0_mi,priori0)
            pc1 = NBG.NBGprob(x_test_std,var1,c1_mi,priori1)            
            res = []
            for j in range(len(pc0)):
                if pc0[j]>pc1[j]:
                    res.append(0)
                else:
                    res.append(1)
            print(f"Summary for the classifier NBG with accuracy {metrics.accuracy_score(y_test,res):.2f}")
            print(metrics.classification_report(y_test, res))
    
    @classmethod
    def NBGprob(cls,indep,var,mi,priori):
        prob_vec = []
        print(f"var:{var.shape}")
        for k in range(indep.shape[0]):
            sum = 0
            for i in range(indep.shape[1]):
                #print((indep[k,[i]][0]-mi[i]))
                sum += np.log(2*np.pi*var) + ((indep[k,[i]][0]-mi[i])**2)/var[i]
            prob_vec.append(-0.5*sum + np.log(priori))
        return prob_vec




            
    


            
    
def main():
    #adg = ADG("teste2.csv",2)
    #adg = ADG("breastcancer.csv",10)
    #adg.ADGstep()
    nbg = NBG("breastcancer.csv",10)
    nbg.NBGstep()
if __name__ == "__main__":
    main()