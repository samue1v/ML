import numpy as np
from OpLib import Operations as op  
from sklearn import tree,metrics

class DecisionTree:
    def __init__(self,file,n_folds):
        self.dataset = np.genfromtxt(file, delimiter=',', skip_header=0)
        np.random.shuffle(self.dataset)
        self.algo = "Decision Tree"
        self.n_folds = n_folds
        self.folds = op.Kfold(self.dataset,self.n_folds)
        self.acuracia = []
        self.revoc = []
        self.precisao = []
        self.f1 = []
    
    def TreeStep(self):
        for i in range(self.n_folds):
            test, train  = op.get_folded_data(self.folds,i)
            x_train,y_train = op.slice_data(train)
            x_test,y_test = op.slice_data(test)
            clf = tree.DecisionTreeClassifier(criterion = 'entropy')
            clf = clf.fit(x_train,y_train)
            y_hat = clf.predict(x_test)
            self.revoc.append(metrics.recall_score(y_test,y_hat))
            self.acuracia.append(metrics.accuracy_score(y_test,y_hat))
            self.precisao.append(metrics.precision_score(y_test,y_hat))
            self.f1.append(metrics.f1_score(y_test,y_hat))

def main():
    dt = DecisionTree("kc2.csv",10)
    dt.TreeStep()
    op.show_score((dt.acuracia),dt.algo,"Acuracia")
    op.show_score((dt.revoc),dt.algo,"Revocação")
    op.show_score((dt.precisao),dt.algo,"precisão")
    op.show_score((dt.f1),dt.algo,"F1-Score")


if __name__ == "__main__":
    main()


