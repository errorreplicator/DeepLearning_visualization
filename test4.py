import pandas as pd
from keras import optimizers
from keras.preprocessing import image
pd.set_option('display.width', 1000)
pd.set_option('display.max_column',None)
pd.set_option('display.max_colwidth', 500)
from matplotlib import pyplot as plt
from models import kerasmodels


PATH = 'c:/dataset/dogbreeds'

ds = pd.read_csv(f'{PATH}/labels.csv')
ds['files'] = [f'{x}.jpg' for x in ds['id']]

breeds_cound = ds.breed.value_counts(ascending=False)

# print(breeds_cound.sort_values(ascending=False))

# ds_small = ds.loc[ds.breed.isin(['scottish_deerhound', 'maltese_dog', 'afghan_hound'])]
print(ds.head(10))
print(ds.shape)
# breeds_cound.plot(kind='bar')
# plt.show()

datagen = image.ImageDataGenerator(rescale=1./255.,
                                   shear_range=0.20,
                                   zoom_range=0.20,
                                   horizontal_flip=True,
                                   vertical_flip=True)
datapipe = datagen.flow_from_dataframe(ds,
                                       directory=f'{PATH}/train/',
                                       x_col='files',
                                       y_col='breed',
                                       target_size=(224,224),
                                       batch_size=32,
                                       color_mode='rgb',
                                       class_mode='categorical')

STEP_SIZE = datapipe.n//datapipe.batch_size
print(STEP_SIZE)
myModel = kerasmodels.AlexNet((224,224,3),120)
# optim = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
# optim = optimizers.SGD(lr=1e-2, momentum=0.9)
# myModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
optim = optimizers.SGD(lr=0.001, momentum=0.99, nesterov=True)
myModel.compile(optimizer=optim,loss='categorical_crossentropy',metrics=['accuracy'])
myModel.fit_generator(datapipe,steps_per_epoch=STEP_SIZE,epochs=20)