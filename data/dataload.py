import cv2
import os
import random

import matplotlib.pyplot as plt

def dogbreeds(dataframe,input_size=3000,resolution=50,test_data=True):
    PATH = 'C:/Dataset/dogbreeds/train/'


def load_data (input_size=3000,resolution=50,test_data=True):

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
    X_test = []
    y_test = []

    random.shuffle(catDog_list)
    index=0
    if test_data==True:
        for x,y in catDog_list:
            index+=1
            if index%10!=0:
                X_train.append(x)
                y_train.append(y)
            else: # every 10th sample is a test sample
                X_test.append(x)
                y_test.append(y)

        print(f'size of {catalogs[0]} train table:',len(X_train))
        print(f'size of {catalogs[1]} train table:',len(y_train))
        print(f'size of {catalogs[0]} test table:', len(X_test))
        print(f'size of {catalogs[1]} test table:', len(y_test))

        return (X_train, y_train,X_test,y_test)
    else:

        for x,y in catDog_list:
            X_train.append(x)
            y_train.append(y)
        print(f'size of {catalogs[0]} table:', y_test.count(0))
        print(f'size of {catalogs[1]} table:', y_test.count(1))
        print(f'Overall table size:', len(y_test))
        print('Please remember to normalize and reshape || simple_norm and simple_reshape')

        return (X_train,y_train)

def load_general_patch (resolution,path,input_size=3000):
    path = path
    catalogs = ['Dog', 'Cat']
    resolution = resolution
    catDog_list = []
    input_size = input_size

    for folder in catalogs:
        index = 0
        fol_path = os.path.join(path, folder)
        for file in os.listdir(fol_path):
            try:
                image = cv2.imread(os.path.join(fol_path, file))  # ,cv2.IMREAD_GRAYSCALE)
                grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            except Exception as e:
                print(f'Error at file index {index} with path:', fol_path, '\\', file, sep='')
                pass

            image_resize = cv2.resize(grey, (resolution, resolution))
            catDog_list.append([image_resize, catalogs.index(folder),file])
            index += 1
            if index > input_size - 1: break
    X_test= []
    y_test = []
    filenames=[]

    random.shuffle(catDog_list)
    for x, y,z in catDog_list:
        X_test.append(x)
        y_test.append(y)
        filenames.append(z)
    print(f'size of {catalogs[0]} table:', y_test.count(0))
    print(f'size of {catalogs[1]} table:', y_test.count(1))
    print(f'Overall table size:', len(y_test))
    print('Please remember to normalize and reshape || simple_norm and simple_reshape')

    return (X_test, y_test,filenames)

def most_correct_dogs(ds,files=False):
    dx = ds[ds['y_test'] == 0.0].sort_values(by=['predictions'], ascending=True)
    if files:
        return dx['index'],list(round(dx['predictions'],4)),dx['filename']
    else:
        return dx['index'], list(round(dx['predictions'], 4))
    #call it like this || general.plots([X_test[x] for x in indexes][:display],titles=predict[:display])

def most_incorrect_dogs(ds,files=False):
    dx = ds[ds['y_test'] == 0.0].sort_values(by=['predictions'], ascending=False)
    if files:
        return dx['index'],list(round(dx['predictions'],4)),dx['filename']
    else:
        return dx['index'], list(round(dx['predictions'], 4))
    #call it like this || general.plots([X_test[x] for x in indexes][:display],titles=predict[:display])

def most_uncertain_dogs(ds,files=False):
    dx = ds[(ds['predictions'] <= 0.5) & (ds['y_test'] == 0.0)].sort_values(by=['predictions'], ascending=False)
    if files:
        return dx['index'],list(round(dx['predictions'],4)),dx['filename']
    else:
        return dx['index'], list(round(dx['predictions'], 4))

def most_correct_cats(ds,files=False):
    dx = ds[ds['y_test'] == 1.0].sort_values(by=['predictions'], ascending=False)
    if files:
        return dx['index'],list(round(dx['predictions'],4)),dx['filename']
    else:
        return dx['index'], list(round(dx['predictions'], 4))

def most_incorrect_cats(ds,files=False):
    dx = ds[ds['y_test'] == 1.0].sort_values(by=['predictions'], ascending=True)
    if files:
        return dx['index'],list(round(dx['predictions'],4)),dx['filename']
    else:
        return dx['index'], list(round(dx['predictions'], 4))

def most_uncertain_cats(ds,files=False):
    dx = ds[(ds['predictions'] >= 0.5) & (ds['y_test'] == 1.0)].sort_values(by=['predictions'], ascending=True)
    if files:
        return dx['index'],list(round(dx['predictions'],4)),dx['filename']
    else:
        return dx['index'], list(round(dx['predictions'], 4))