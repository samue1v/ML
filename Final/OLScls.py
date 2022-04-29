from LinearReg import LinearReg
import numpy as np
import matplotlib.pyplot as plt
class OLS(LinearReg):
    def __init__(self,file,test_pct):
        super().__init__(file,test_pct)

    def OLS_step(self,label):
        self.xo = self.get_x()["x_"+label]
        self.x = self.get_x(True)["x_"+label]
        self.y = self.get_y()["y_"+label]
        self.w = (np.linalg.inv(self.x.T @ self.x) @ self.x.T) @ self.y
        self.y_hat = self.x @ self.w
        self.rmse = np.sqrt((np.mean((self.y-self.y_hat)**2)))
    def OLSshow(self):
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.scatter(self.xo,self.y)
        plt.plot(self.xo,self.y_hat)
        plt.gcf().canvas.set_window_title('OLS')
        plt.show()


    

def main():
    l = OLS("california.csv",0.8)
    l.OLS_step("test")
    print(l.w)
    #l.OLSshow()
    
if __name__ == "__main__":
    main()