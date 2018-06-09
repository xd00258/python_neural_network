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
#np.random.seed(1)

#weight matrix of dimension 3x1 initialized at random, mean of 0 between -1 and 1
syn0 = 2*np.random.random((3,1)) - 1

#begin training
for iter in range(10000):

    #forward propagation
    l0 = X         #layer 0 is our input dataset with 4 training examples
    l1 = nonlin(np.dot(l0,syn0))    #layer 1 is our prediction step

    #calculate miss/loss
    l1_error = y - l1
    l1_delta = l1_error * nonlin(l1,True)

    #update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output after training: ")
print(l1)
