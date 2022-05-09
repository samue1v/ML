import matplotlib.pyplot as plt
import numpy as np
from LinearReg import LinearRegcls
from OperationsLib import Operations as op

class GD(LinearRegcls):
    def __init__(self, file,pace,it_num,test_pct=1):
        super().__init__(file, test_pct)
        self.pace = pace
        self.it_num = it_num
        self.it_lst = []
        self.mse_list = []
    def GDstep(self):
        t = 0
        #print("y:")
        y_n = op.stddize(self.y_test)
        #print("x:")
        x_n = op.stddize(self.x_test)#nxd
        x_co = op.concat_one_column(x_n)#nxd+1
        self.w = np.zeros((x_co.shape[1],1))#d+1x1
        #print(self.w)
        while t< self.it_num:
            t+=1
            self.it_lst.append(t)
            y_hat = x_co @ self.w      #np.array([np.sum(self.w * x_co,axis=1)]).T
            #print(f'Y-hat {y_hat}')
            err = y_n - y_hat
            print(self.w)
            self.w = self.w + self.pace*(np.array([np.mean(err*x_co,axis=0)])).T
            y_hot = op.de_stddize(self.y_test,y_hat)
            e = self.y_test - y_hot
            mse = np.mean(e**2)
            self.mse_list.append(mse)
        
        plt.subplot(1,2,1)
        plt.ylabel("MSE")
        plt.xlabel("Iterations")
        plt.plot(self.it_lst,self.mse_list)


        plt.subplot(1,2,2)
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.scatter(self.x_test,self.y_test)

        #plt.plot(self.x_test,np.sum((self.w*op.concat_one_column(self.x_test)),axis=1))
        plt.plot(self.x_test,y_hot)
        


        plt.gcf().canvas.set_window_title('GD')

        plt.show()




def main():
    newGD= GD("artificial1d.csv",0.1,100)
    #newGD= GD("teste2.csv",0.1,10)
    newGD.GDstep()

if __name__ == "__main__":
    main()
        
