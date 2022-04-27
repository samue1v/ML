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