import numpy as np

class Operations:
    @staticmethod
    def stddize(matrix):
        one_column = np.ones((matrix.shape[0],1))
        inv_stddev = np.divide(np.ones((matrix.shape[1],matrix.shape[1])), (matrix.std(axis=0))) * (np.identity(matrix.shape[1]))
        mean = (matrix.mean(axis=0))*(np.ones((matrix.shape[0],matrix.shape[1])))
        m_std = (matrix-mean)@(inv_stddev)
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

    @staticmethod
    def get_orderN_matrix(matrix,order):
        n_order_matrix = matrix
        for i in range(2,order+1):
            n_order_matrix = np.concatenate((n_order_matrix,matrix**i),axis=1)
        return n_order_matrix

    @staticmethod
    def suhffleEntireData(dataset):
        return np.random.shuffle(dataset)
    
    @staticmethod
    def shuffleIndepData(dataset):
        np.random.shuffle(dataset[:,0:-2])
