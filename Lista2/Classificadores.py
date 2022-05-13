import numpy as np
from OperationsLib import Operations as op
from sklearn import metrics
class Classificadores():
    def __init__(self,file,n_folds):
        self.dataset = np.genfromtxt(file, delimiter=',', skip_header=0)
        self.n_folds = n_folds
        self.folds = op.Kfold(self.dataset,self.n_folds)
        self.acuracia = []
        self.revoc = []
        self.precisao = []
        self.f1 = []

class ADG(Classificadores):
    def __init__(self,file,n_folds):
        super().__init__(file,n_folds)
        self.algo = "ADG"
        
    
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
            priori0 = np.count_nonzero(y_train == 0)/y_train.shape[0]
            priori1 = 1-priori0
            pc0 = ADG.ADGprob(x_test_std,m_cov0,c0_mi,priori0)
            pc1 = ADG.ADGprob(x_test_std,m_cov1,c1_mi,priori1)            
            res = []
            for j in range(len(pc0)):
                if pc0[j]>pc1[j]:
                    res.append(0)
                else:
                    res.append(1)
            self.revoc.append(metrics.recall_score(y_test,res))
            self.acuracia.append(metrics.accuracy_score(y_test,res))
            self.precisao.append(metrics.precision_score(y_test,res))
            self.f1.append(metrics.f1_score(y_test,res))
    
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
        self.algo = "NBG"
    
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
            var0 = np.var(indepc0,axis=0)
            var1 = np.var(indepc1,axis=0)
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
            self.revoc.append(metrics.recall_score(y_test,res))
            self.acuracia.append(metrics.accuracy_score(y_test,res))
            self.precisao.append(metrics.precision_score(y_test,res))
            self.f1.append(metrics.f1_score(y_test,res))
    
    @classmethod
    def NBGprob(cls,indep,var,mi,priori):
        prob_vec = []
        for k in range(indep.shape[0]):
            sum = 0
            for i in range(indep.shape[1]):
                sum += np.log(2*np.pi*var[i]) + ((indep[k][i]-mi[i])**2)/var[i]
            prob_vec.append(-0.5*sum + np.log(priori))
        return prob_vec

    
def main():
    adg = ADG("breastcancer.csv",10)
    adg.ADGstep()
    op.show_score((adg.acuracia),adg.algo,"Acuracia")
    op.show_score((adg.revoc),adg.algo,"Revocação")
    op.show_score((adg.precisao),adg.algo,"precisão")
    op.show_score((adg.f1),adg.algo,"F1-Score")
    nbg = NBG("breastcancer.csv",10)

    nbg.NBGstep()
    op.show_score(nbg.acuracia,nbg.algo,"Acuracia")
    op.show_score(nbg.revoc,nbg.algo,"Revocação")
    op.show_score(nbg.precisao,nbg.algo,"precisão")
    op.show_score(nbg.f1,nbg.algo,"F1-Score")
if __name__ == "__main__":
    main()