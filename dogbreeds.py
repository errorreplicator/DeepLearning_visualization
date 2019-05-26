# import pandas as pd
from data import dataload
from tasking import general
from models import kerasmodels
import keras
from keras.applications import VGG16
from keras.layers import Dense
from keras import models
import numpy as np

X_train, y_train = dataload.dogbreedsD(10222,resolution=224,test_data=False)
X_train = general.simple_reshape_color(X_train,224)
X_train = general.simple_norm(X_train)

## X_test = general.simple_reshape(X_test,64)
## X_test = general.simple_norm(X_test)

# model = kerasmodels.dogbreedsM((64,64,1),120)
# model.fit(X_train,y_train,validation_split=0.1,epochs=10,batch_size=32)

model = VGG16()#, include_top=False)
model.layers.pop()
myModel = keras.models.Sequential()

for layer in model.layers:
    myModel.add(layer)

for layer in myModel.layers:
    myModel.trainable = False

myModel.add(Dense(120,activation='softmax'))

myModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
myModel.fit(X_train,y_train,batch_size=16,epochs=10)

