from LinearReg import LinearReg
import numpy as np
import matplotlib.pyplot as plt
class GD(LinearReg):
    def __init__(self,file,test_pct,alpha,it_num):
        super().__init__(file,test_pct)
        self.alpha = alpha
        self.it_num = it_num
    def GDstep(self):
        self.it_list = []
        self.mseLst = []
        self.x = self.get_x(True)["x_test"]
        self.xo =self.get_x()["x_test"]
        self.y = self.get_y()["y_test"]
        y_std = LinearReg.stddize(self.y,False)
        x_std = LinearReg.stddize(self.xo,True)
        self.w=np.zeros((1,x_std.shape[1]))#parameters initial values
        t=0
        while t < self.it_num:
            t+=1
            self.it_list.append(t) 
            self.y_hat = np.sum((x_std * self.w).T, axis=0)
            self.y_hat = np.array([self.y_hat]).T #y_hat
            e_std = y_std-self.y_hat#error
            m_err = e_std * x_std #ei x xi
            k = m_err.mean(axis=0)*self.alpha #Gradient
            self.w = self.w + k
            self.y_hato = LinearReg.de_stddize(self.y,self.y_hat)
            e = (self.y-self.y_hato) 
            mse = (np.mean(e**2))
            self.mseLst.append(mse)
    
    def GDshow(self):
        plt.subplot(1,2,1)
        plt.ylabel("MSE")
        plt.xlabel("Iterations")
        plt.plot(self.it_list,self.mseLst)


        plt.subplot(1,2,2)
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.scatter(self.xo,self.y)
        plt.plot(self.xo,self.y_hato)

        plt.gcf().canvas.set_window_title('GD')

        plt.show()

    
def main():
    g = GD("artificial1d.csv",1,0.01,1000)
    g.GDstep()
    g.GDshow()

if __name__ == "__main__":
    main()
        

    