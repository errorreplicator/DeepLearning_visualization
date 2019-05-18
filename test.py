from matplotlib import pyplot as plt
import numpy as np
np.random.seed(99)

and_in = [[0,0],
          [0,1],
          [1,0],
          [1,1]]

y = [0,0,0,1]

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

X = np.array(and_in)
w1 = np.random.randn(2,3)
w2 = np.random.randn(3,1)

def forwardprop(w1,w2,x):
    z2 = np.dot(x,w1)
    a2 = sigmoid(z2)
    z3 = np.dot(a2,w2)
    y_hat = sigmoid(z3)
    return z2, a2,z3,y_hat

def sigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

def cost(y,y_hat):
    return 0.5*sum((y-y_hat)**2)

def costFunctionPrime(X, y,w1,w2):
    # Compute derivative with respect to W and W2 for a given X and y:
    z2, a2, z3, y_hat = forwardprop(w1,w2,X)
    delta3 = np.multiply(-(y - y_hat), sigmoidPrime(z3))
    print('delta3',delta3.shape)
    dJdW2 = np.dot(a2.T, delta3)
    print('a2',a2.T.shape)
    delta2 = np.dot(delta3, w2.T) #* sigmoidPrime(z2)
    # print(delta2.shape)

    # dJdW1 = np.dot(X.T, delta2)
    #
    # return dJdW1, dJdW2, y_hat

scalar = 0.2
costFunctionPrime(X,y,w1,w2)
# for x in range (100):
#     dJdW1, dJdW2, y_hat = costFunctionPrime(X,y,w1,w2)
#     print(cost(y,y_hat))
#     w1 = w1 - scalar*dJdW1
#     w2 = w2 - scalar*dJdW2
#     if x%10==0:
#         print(y_hat)
# print(w2)




