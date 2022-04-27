import numpy as np
class LinearReg:
    def __init__(self,file,test_pct=1.0):
        self.file = file
        self.test_pct = test_pct
        self.dataset = np.genfromtxt(file, delimiter=',', skip_header=0)
        if test_pct < 1.0:
            np.random.shuffle(dataset)

    @classmethod
    def stddize(cls,matrix,has_one_column):
        one_column = np.ones((matrix.shape[0],1))
        inv_stddev = np.divide(np.ones((matrix.shape[1],matrix.shape[1])), (matrix.std(axis=0))) * (np.identity(matrix.shape[1]))
        mean = (matrix.mean(axis=0))*(np.ones((matrix.shape[0],matrix.shape[1])))
        x_std = (matrix-mean)@(inv_stddev)
        if(has_one_column):
            return np.concatenate((one_column,x_std),axis = 1)
        return x_std

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
    

       

class OLS(LinearReg):
    def __init__(self,file,range_columns):
        super().__init__(file,range_columns)

    def OLS_step(self):
        x = self.get_x(True)["x_test"]
        y = self.get_y()["y_test"]
        w = (np.linalg.inv(x.T @ x) @ x.T) @ y
        y_hat = x @ w
        mse = (np.mean((y-y_hat)**2))
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.scatter(x[:,[1]],y)
        plt.plot(x[:,[1]],y_hat)

        plt.gcf().canvas.set_window_title('OLS')

        plt.show()


    

def main():
    l = OLS("artificial1d.csv",1)
    l.OLS_step()
    
if __name__ == "__main__":
    main()