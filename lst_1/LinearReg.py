import numpy as np
class LinearReg():
    def __init__(self,file,test_pct=1.0):
        super().__init__()
        self.file = file
        self.test_pct = test_pct
        self.dataset = np.genfromtxt(file, delimiter=',', skip_header=0)
        if test_pct < 1.0:
            np.random.shuffle(dataset)

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



    def get_x(self, has_one_column : bool = False):
        x_test = (self.dataset[0:int((self.dataset.shape[0])*self.test_pct), :])[:, 0:int(self.dataset.shape[1]-1)] 
        x_training = (self.dataset[int((self.dataset.shape[0])*self.test_pct):, :])[:, 0:int(self.dataset.shape[1]-1)] 
        if(has_one_column):
            test_one_vec = np.ones((x_test.shape[0],1))
            x_test = np.concatenate((test_one_vec,x_test),axis = 1)
            train_one_vec = np.ones((x_training.shape[0],1))
            x_training = np.concatenate((train_one_vec,x_training),axis = 1)
        return {"x_test":x_test,"x_training":x_training}
    
    def get_y(self):
        y_test = (self.dataset[0:int((self.dataset.shape[0])*self.test_pct), :])[:, [-1]] 
        y_training = (self.dataset[int((self.dataset.shape[0])*self.test_pct):, :])[:, [-1]] 
        return {"y_test":y_test,"y_training":y_training}
    


#falta de_stddize, Graphs.py, checar se o gd functiona com std data
# e a 2 questao. 
    