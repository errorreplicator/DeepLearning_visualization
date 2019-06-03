import keras
from keras.preprocessing import image
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

PATH = 'C:/Dataset/dogbreeds/'

dataframe = pd.read_csv(f'{PATH}labels.csv')
dataframe['path'] = [f'{PATH}train/{x}.jpg' for x in dataframe['id']]

img2 = image.load_img('C:/Dataset/dogbreeds/train/003df8b8a8b05244b1d920bb6cf451f9.jpg')
img2 = image.img_to_array(img2)
plt.imshow(img2/255)

index = 0
img_batch = []
for inx,frame in dataframe.iterrows():
    img = image.load_img(frame['path'])
    img = image.img_to_array(img)/255
    img_batch.append(img)
    print('appending')
    if index == 9:
        break
    index+=1


fig, axs = plt.subplots(3, 3, figsize=(12, 12))
axs = axs.flatten()
for img, ax in zip(img_batch, axs):
    ax.imshow(img)
plt.show()
