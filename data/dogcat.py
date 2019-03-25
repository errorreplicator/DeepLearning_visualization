import cv2
import os
import random

import matplotlib.pyplot as plt

def load_data (input_size,resolution):

    path = 'C:\Dataset\img'
    catalogs = ['Dog','Cat']
    resolution = resolution
    input_size = input_size
    catDog_list = []

    for folder in catalogs:
        index = 0
        fol_path = os.path.join(path,folder)
        for file in os.listdir(fol_path):
            try:
                image = cv2.imread(os.path.join(fol_path, file))  # ,cv2.IMREAD_GRAYSCALE)
                grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            except Exception as e:
                print(f'Error at file index {index} with path:', fol_path, '\\', file, sep='')
                pass

            image_resize = cv2.resize(grey, (resolution, resolution))
            catDog_list.append([image_resize, catalogs.index(folder)])
            index+=1
            if index > input_size-1: break
    X_train = []
    y_train = []

    random.shuffle(catDog_list)

    for x,y in catDog_list:
        X_train.append(x)
        y_train.append(y)
    print(f'size of {catalogs[0]} table:',len(X_train))
    print(f'size of {catalogs[1]} table:',len(y_train))

    return (X_train,y_train)