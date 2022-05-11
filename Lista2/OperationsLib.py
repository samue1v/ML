import numpy as np

class Operations:


    #@staticmethod
    #def calc_pct(pred,comp):



    @staticmethod
    def sum_covm(x,mi):
        res = np.zeros((mi.shape[0],mi.shape[0]))
        
        for i in range(x.shape[0]):
            vec = x[i] - mi
            res += np.array([vec]).T @ np.array([vec])
        return res/(x.shape[1]-1)

    @staticmethod
    def slice_data(data):
        x = data[: , 0:-1]
        y= data[:, -1:]
        return x,y
    
    @staticmethod
    def get_folded_data(folds,n):
        cp = folds.copy()
        test = cp.pop(n)
        train = np.vstack(cp)
        return test,train


    @staticmethod
    def Kfold(dataset,n_folds):
        fold_lst = []
        fold_size = int(dataset.shape[0]/n_folds)
        fold_remainder = (dataset.shape[0]%n_folds)
        for i in range(n_folds):
            floor = i*fold_size
            fold_lst.append(dataset[floor:floor+fold_size])
        for j in range(fold_remainder):
            fold_lst[j] = np.r_[fold_lst[j],np.array(dataset[-j-2:-j-1,:])]
        return fold_lst

    @staticmethod
    def stddize(matrix):
        mean = (matrix.mean(axis=0))*(np.ones((matrix.shape[0],matrix.shape[1])))
        m_std = np.divide((matrix-mean),(matrix.std(axis=0)))
        return m_std

    @staticmethod
    def de_stddize(vec,vec_hat):
        std_dev_vec = (vec.std(axis=0))
        mean_vec = (vec.mean(axis=0))*(np.ones((vec.shape[0],vec.shape[1])))
        return (std_dev_vec*vec_hat) + mean_vec

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    
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
        np.random.shuffle(dataset)
    
    @staticmethod
    def shuffleIndepData(dataset):
        np.random.shuffle(dataset[:,0:-1])
