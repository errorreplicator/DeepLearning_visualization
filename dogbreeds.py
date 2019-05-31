# myModel = multi_gpu_model(myModel,gpus=2) --- paralize computing
# import pandas as pd
from data import dataload
from tasking import general
from models import kerasmodels
import keras
from keras.applications import VGG16
from keras.layers import Dense
# from keras import models
import numpy as np
from keras import optimizers

X_train, y_train = dataload.dogbreedsD(10,resolution=224,test_data=False)
print(np.array(X_train).shape)
X_train = general.simple_reshape_color(X_train,224)
X_train = general.simple_norm(X_train)

## X_test = general.simple_reshape(X_test,64)
## X_test = general.simple_norm(X_test)

# model = kerasmodels.dogbreedsM((64,64,1),120)
# model.fit(X_train,y_train,validation_split=0.1,epochs=10,batch_size=32)



model = VGG16(weights='imagenet')#, include_top=False)
model.layers.pop()
myModel = keras.models.Sequential()

for layer in model.layers:
    myModel.add(layer)

for layer in myModel.layers:
    myModel.trainable = False

myModel.add(keras.layers.Dense(120,activation='softmax'))

# myModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

myModel.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy']) #<- actually works with train data - -- momentum ??


# loss: 0.0292 - acc: 0.9952 - val_loss: 1.9108 - val_acc: 0.6207 - top layer ON



myModel.fit(X_train,y_train,batch_size=16,epochs=10)

myModel.fit(X_train,y_train,batch_size=16,epochs=30,validation_split=0.1)



