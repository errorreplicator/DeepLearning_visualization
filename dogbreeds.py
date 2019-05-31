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


# loss: 0.0292 - acc: 0.9952 - val_loss: 1.9108 - val_acc: 0.6207 - top layer ON



myModel.fit(X_train,y_train,batch_size=16,epochs=10)

myModel.fit(X_train,y_train,batch_size=16,epochs=30,validation_split=0.1)


# Epoch 1/10
# 9199/9199 [=========] - 95s 10ms/step - loss: 4.8902 - acc: 0.0118 - val_loss: 4.7450 - val_acc: 0.0186
# Epoch 2/10
# 9199/9199 [=========] - 86s 9ms/step - loss: 4.5778 - acc: 0.0369 - val_loss: 4.2352 - val_acc: 0.0733
# Epoch 3/10
# 9199/9199 [=========] - 86s 9ms/step - loss: 3.6058 - acc: 0.1572 - val_loss: 2.6060 - val_acc: 0.3363
# Epoch 4/10
# 9199/9199 [=========] - 86s 9ms/step - loss: 2.3150 - acc: 0.3823 - val_loss: 1.8662 - val_acc: 0.4858
# Epoch 5/10
# 9199/9199 [=========] - 85s 9ms/step - loss: 1.6772 - acc: 0.5280 - val_loss: 1.5373 - val_acc: 0.5601
# Epoch 6/10
# 9199/9199 [=========] - 85s 9ms/step - loss: 1.3286 - acc: 0.6198 - val_loss: 1.4312 - val_acc: 0.5885
# Epoch 7/10
# 9199/9199 [=========] - 85s 9ms/step - loss: 1.0866 - acc: 0.6811 - val_loss: 1.3890 - val_acc: 0.5943
# Epoch 8/10
# 9199/9199 [=========] - 86s 9ms/step - loss: 0.9123 - acc: 0.7289 - val_loss: 1.3578 - val_acc: 0.6041
# Epoch 9/10
# 9199/9199 [=========] - 87s 9ms/step - loss: 0.7625 - acc: 0.7663 - val_loss: 1.2653 - val_acc: 0.6276
# Epoch 10/10
# 9199/9199 [=========] - 86s 9ms/step - loss: 0.6131 - acc: 0.8095 - val_loss: 1.3156 - val_acc: 0.6246
#
# Epoch 1/5
# 9199/9199 [=========] - 84s 9ms/step - loss: 0.5161 - acc: 0.8420 - val_loss: 1.3246 - val_acc: 0.6129
# Epoch 2/5
# 9199/9199 [=========] - 85s 9ms/step - loss: 0.4071 - acc: 0.8714 - val_loss: 1.3870 - val_acc: 0.6295
# Epoch 3/5
# 9199/9199 [=========] - 85s 9ms/step - loss: 0.3360 - acc: 0.8981 - val_loss: 1.3809 - val_acc: 0.6383
# Epoch 4/5
# 9199/9199 [=========] - 84s 9ms/step - loss: 0.2680 - acc: 0.9184 - val_loss: 1.3755 - val_acc: 0.6413
# Epoch 5/5
# 9199/9199 [=========] - 85s 9ms/step - loss: 0.2056 - acc: 0.9360 - val_loss: 1.4541 - val_acc: 0.6364