from tensorflow.contrib import keras
import time
import datetime
from tensorflow.contrib.keras import models

km = keras.models
kl = keras.layers

def retTime():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    return st


def modelSeq1(input_shape):

    model = km.Sequential()

    model.add(kl.Conv2D(128,(3,3),input_shape=input_shape))
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPool2D(pool_size=(2,2)))

    model.add(kl.Conv2D(256,(3,3)))
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPool2D(2,2))

    model.add(kl.Flatten())

    model.add(kl.Dense(64))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(1))
    model.add(kl.Activation('sigmoid'))

    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model

def saveModelAll(model,filename):
    model.save(f'repo/{filename}.{retTime()}.h5')

def loadModel(path):
    model = models.load_model(path)
    return model

def loadWeights(path):
    pass

def saveWeights(weights):
    pass