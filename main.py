import numpy as np

#nonlinearity mapping sigmoid function
def nonlin(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input dataset
X = np.array([
    [0,0,1],    #one row is one training example
    [0,1,1],    #columns are input node
    [1,0,1],
    [1,1,1]
    ])

#output dataset
y = np.array([[0,0,1,1]]).T

#seed numbers for always same random sequence (for practice)
np.random.seed(1)

#weight matrix of dimension 3x1 initialized at random
syn0 = 2*np.random.random((3,1)) - 1




print(syn0)