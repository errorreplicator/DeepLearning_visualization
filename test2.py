import keras
from models import kerasmodels
from keras.datasets import mnist
from matplotlib import pyplot as plt

(X_train,y_train),(x_test,y_test) = mnist.load_data()

print(type(X_train))
print(X_train.shape)
print(y_train.shape)
# print(y_train[:5])
print(X_train[0].shape)

for x in range(10):
    plt.imshow(X_train[x],cmap=plt.cm.binary)
    plt.show()