from LinearReg import LinearReg
import numpy as np
import matplotlib.pyplot as plt
class OLS(LinearReg):
    def __init__(self,file,test_pct):
        super().__init__(file,test_pct)

    def OLS_step(self):
        self.xo = self.get_x()["x_test"]
        self.x = self.get_x(True)["x_test"]
        self.y = self.get_y()["y_test"]
        self.w = (np.linalg.inv(self.x.T @ self.x) @ self.x.T) @ self.y
        self.y_hat = self.x @ self.w
        self.mse = (np.mean((self.y-self.y_hat)**2))
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.scatter(self.xo,self.y)
        plt.plot(self.xo,self.y_hat)
        plt.gcf().canvas.set_window_title('OLS')
        plt.show()


    

def main():
    l = OLS("artificial1d.csv",1)
    l.OLS_step()
    
if __name__ == "__main__":
    main()