import numpy as np

#nonlinearity mapping sigmoid function
def nonlin(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))



#print (nonlin(5,True))