from LinearReg import LinearReg
import numpy as np
import matplotlib.pyplot as plt
class GDS(LinearReg):
    def __init__(self,file,test_pct,alpha,it_num):
        super().__init__(file,test_pct)
        self.alpha = alpha
        self.it_num = it_num
    def GDSstep(self):
        self.it_list = []
        self.mseLst = []
        self.x = self.get_x(True)["x_test"]
        self.xo =self.get_x()["x_test"]
        self.y = self.get_y()["y_test"]
        y_std = LinearReg.stddize(self.y,False)
        x_std = LinearReg.stddize(self.xo,True)
        self.w=np.zeros((1,x_std.shape[1]))#parameters initial values
        print(self.w.shape)
        t=0
        while t < self.it_num:
            t+=1
            self.it_list.append(t)
            np.random.shuffle(x_std)
            #print(x_std)
            for i in range(self.x.shape[1]):
                #print(x_std.shape)
                #print(self.w.shape)
                self.y_hat = np.sum(x_std[[i],:] * self.w)
                #self.y_hat = np.array([self.y_hat]).T #y_hat
                e_std = y_std[[i],:]-self.y_hat#error
                print(y_std[[i],:].shape)
                m_err = e_std * x_std[[i],:] * self.alpha #ei x xi
                #k = m_err.mean(axis=0)*self.alpha #Gradient
                self.w = self.w + m_err
                self.y_hato = LinearReg.de_stddize(self.y,self.y_hat)
                e = (self.y-self.y_hato) 
                mse = (np.mean(e**2))
                self.mseLst.append(mse)
    
    def GDSshow(self):
        #plt.subplot(1,2,1)
        #plt.ylabel("MSE")
        #plt.xlabel("Iterations")
        #plt.plot(self.it_list,self.mseLst)


        #plt.subplot(1,2,2)
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.scatter(self.xo,self.y)
        plt.plot(self.xo,self.y_hato)

        plt.gcf().canvas.set_window_title('GD')

        plt.show()

    
def main():
    g = GDS("artificial1d.csv",1,0.01,1000)
    g.GDSstep()
    g.GDSshow()

if __name__ == "__main__":
    main()