import numpy as np
b=np.array(0)

a =np.ones((5,5))
print(np.concatenate((a,b),axis=0))