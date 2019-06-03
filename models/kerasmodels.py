from tensorflow.contrib.keras import optimizers
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np

km = keras.models
kl = keras.layers


def AlexNet(classnum):

    np.random.seed(1000)
    # Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(classnum))
    model.add(Activation('softmax'))

    return model

def dogbreedsM(input_shape,classes):

    model = km.Sequential()

    model.add(kl.Conv2D(128,(3,3),input_shape=input_shape))
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPool2D(pool_size=(2,2)))

    model.add(kl.Conv2D(256,(3,3),name='forVisual'))
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPool2D(2,2))

    model.add(kl.Flatten())

    model.add(kl.Dense(64))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(classes))
    model.add(kl.Activation('softmax'))

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def modelSeq1(input_shape,classes):

    model = km.Sequential()

    model.add(kl.Conv2D(128,(3,3),input_shape=input_shape))
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPool2D(pool_size=(2,2)))

    model.add(kl.Conv2D(256,(3,3),name='forVisual'))
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPool2D(2,2))

    model.add(kl.Flatten())

    model.add(kl.Dense(64))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(classes))
    model.add(kl.Activation('softmax'))

    # model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model

def modelLeNet(input_shape,classes): #LeNet model
    # Initialize the model
    model =km.Sequential()

    # The first set of CONV => RELU => POOL layers
    model.add(kl.Conv2D(20, (5, 5), padding="same",
                     input_shape=input_shape))
    model.add(kl.Activation("relu"))
    model.add(kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(kl.Dropout(0.2))

    # The second set of CONV => RELU => POOL layers
    model.add(kl.Conv2D(50, (5, 5), padding="same",name='forVisual'))
    model.add(kl.Activation("relu"))
    model.add(kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(kl.Dropout(0.2))

    # The set of FC => RELU layers
    model.add(kl.Flatten())
    model.add(kl.Dense(500))
    model.add(kl.Activation("relu"))
    model.add(kl.Dropout(0.1))

    # The softmax classifier
    model.add(kl.Dense(classes))
    model.add(kl.Activation("sigmoid")) #SOFTMAX in original text

    # If a weights path is supplied, then load the weights
    # if weightsPath is not None:
    #     model.load_weights(weightsPath)
    opt = optimizers.SGD(lr=0.01)
    model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["accuracy"])
    return model
