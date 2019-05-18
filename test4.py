import numpy as np
np.random.seed(1)
x = np.array([[0,0],
          [0,1],
          [1,0],
          [1,1]])

y = np.array([0,0,0,1]).reshape(-1,1)

np.random.seed(99)

w1 = np.random.randn(2,3)
w2 = np.random.randn(3,1)
# print(w1)
# print()
# print(w2)
def softmax(z):
    return 1/(1+np.exp(-z))
def relu(z):
    return np.maximum(0,z)

def loss(y,yHat): #sume of square errors
    # print('y',len(y))
    # print('Yh',yHat,yHat.shape)
    return 0.5*sum((y-yHat)**2)

def nloss(y,Yh): #Cross-Entropy Loss Function
    loss = (1. / x.shape[0]) * (-np.dot(y, np.log(Yh).T) - np.dot(1 - y, np.log(1 - Yh).T))
    return loss

def sigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

def forward(x,w1,w2):
    z2 = np.dot(x,w1)
    a2 = softmax(z2)
    z3 = np.dot(a2,w2)
    yHat = softmax(z3)
    # print(yHat)
    return yHat, z3, a2, z2

def backprop(x,w1,w2):
    yHat, z3, a2, z2 = forward(x,w1,w2)
    # print('-(y - yHat)',(y - yHat).shape)
    # print('sigmoidPrime(z3)',sigmoidPrime(z3).shape)
    z3Hat = np.multiply(-(y - yHat), sigmoidPrime(z3))
    dJdW2 = np.dot(a2.T, z3Hat)
    z2Hat = np.dot(z3Hat, w2.T) * sigmoidPrime(z2)
    dJdW1 = np.dot(x.T, z2Hat)
    return dJdW1, dJdW2, yHat

scalar = 0.5
def scors(y):
    return np.around(y)


def trainNN(epoch,x,w1,w2):
    for i in range (epoch):
        dJdW1, dJdW2, yHat = backprop(x,w1,w2)
        w1 = w1 - scalar*dJdW1
        w2 = w2 - scalar*dJdW2
        if i%100==0:
            print('epoch',i)
            print('yHat: ',yHat)
            print('loss: ',loss(y,yHat))
            print(5*'\n')
    score = scors(yHat)
    return score


score = trainNN(2000,x,w1,w2)
print(2 * '*****\n')
print(score)




