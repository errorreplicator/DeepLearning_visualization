from keras.datasets import mnist
import numpy as np
(Xtrain,ytrain),(Xtest,ytest)= mnist.load_data()

print(Xtrain[0].shape)

dump = Xtrain[0]
np.savetxt('C:/PythonProj/processing/tabe.csv',dump,delimiter=',')