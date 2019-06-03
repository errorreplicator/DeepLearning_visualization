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
fc2 = model.layers[-2]
fc1 = model.layers[-3]
model.layers.pop()

myModel = keras.models.Sequential()
for layer in model.layers[:-2]:
    myModel.add(layer)
myModel.add(fc1)
myModel.add(keras.layers.Dropout(.4))
myModel.add(fc2)
myModel.add(keras.layers.Dropout(.3))

for layer in myModel.layers:
    myModel.trainable = False

myModel.add(keras.layers.Dense(120, activation='softmax'))



# myModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

myModel.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy']) #<- actually works with train data - -- momentum ??