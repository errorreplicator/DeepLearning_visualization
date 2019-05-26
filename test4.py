from data import dataload
from tasking import general
import numpy as np


X_train, y_train = dataload.dogbreedsD(5,resolution=224,test_data=False)

print(type(X_train))
X_train = np.array(X_train)
print(X_train.shape)
print(X_train[0].shape)
X_train = general.simple_reshape_color(X_train,224)
print()
print(X_train.shape)
print(X_train[0].shape)
print()
X_2nd, y_2nd = dataload.load_data(input_size=5,test_data=False)

X_2nd = np.array(X_2nd)
print(X_2nd.shape)
print(X_2nd[0].shape)
X_2nd = general.simple_reshape(X_2nd,50)
print()
print(X_2nd.shape)
print(X_2nd[0].shape)

# X_train = general.simple_reshape_color(X_train,224)
# X_train = general.simple_norm(X_train)

# print(X_train[0])
# print(20*'*')
print(X_2nd[0])

X_train = general.simple_norm(X_train)
X_2nd = general.simple_norm(X_2nd)


print(X_train[0])
print(200*'*')
print(X_2nd[0])