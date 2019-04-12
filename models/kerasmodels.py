from tensorflow.contrib import keras
from tensorflow.contrib.keras import optimizers

km = keras.models
kl = keras.layers



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
    model.add(kl.Activation('sigmoid'))

    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model

def modelLeNet(input_shape,classes): #LeNet model
    # Initialize the model
    model =km.Sequential()

    # The first set of CONV => RELU => POOL layers
    model.add(kl.Conv2D(20, (5, 5), padding="same",
                     input_shape=input_shape))
    model.add(kl.Activation("relu"))
    model.add(kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # The second set of CONV => RELU => POOL layers
    model.add(kl.Conv2D(50, (5, 5), padding="same",name='forVisual'))
    model.add(kl.Activation("relu"))
    model.add(kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # The set of FC => RELU layers
    model.add(kl.Flatten())
    model.add(kl.Dense(500))
    model.add(kl.Activation("relu"))

    # The softmax classifier
    model.add(kl.Dense(classes))
    model.add(kl.Activation("sigmoid")) #SOFTMAX in original text

    # If a weights path is supplied, then load the weights
    # if weightsPath is not None:
    #     model.load_weights(weightsPath)
    opt = optimizers.SGD(lr=0.01)
    model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["accuracy"])
    return model
