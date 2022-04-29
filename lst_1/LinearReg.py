import numpy as np
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
        

    @staticmethod
    def stddize(matrix,has_one_column : bool):
        one_column = np.ones((matrix.shape[0],1))
        inv_stddev = np.divide(np.ones((matrix.shape[1],matrix.shape[1])), (matrix.std(axis=0))) * (np.identity(matrix.shape[1]))
        mean = (matrix.mean(axis=0))*(np.ones((matrix.shape[0],matrix.shape[1])))
        m_std = (matrix-mean)@(inv_stddev)
        if(has_one_column):
            return np.concatenate((one_column,m_std),axis = 1)
        return m_std

    @staticmethod
    def de_stddize(vec,vec_hat):
        std_dev_vec = (vec.std(axis=0))
        mean_vec = (vec.mean(axis=0))*(np.ones((vec.shape[0],vec.shape[1])))
        return (std_dev_vec*vec_hat) + mean_vec

    
    @staticmethod
    def concat_one_column(matrix):
        one_vec = np.ones((matrix.shape[0],1))
        return np.concatenate((one_vec,matrix),axis = 1)

    
    def get_xTraining(self, has_one_column : bool = False):
        if(has_one_column):
            self.x_training = LinearReg.concat_one_column(self.x_training)
        return self.x_training


    def get_xTest(self, has_one_column : bool = False):
        if(has_one_column):
            self.x_test = LinearReg.concat_one_column(self.x_test)
        return self.x_test

    def get_yTest(self):
        return self.y_test
    
    def get_yTraining(self):
        return self.y_training

    def get_orderN_matrix(self,matrix,order):
        for i in range(1,order+1):
            np.concatenate(matrix,matrix**i,axis=1)

    


#falta de_stddize, Graphs.py, checar se o gd functiona com std data
# e a 2 questao. 
    