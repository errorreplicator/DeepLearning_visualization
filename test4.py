
import pandas as pd
import numpy as np

input_size = 3000
resolution = 224
test_data = True


PATH = 'c:/Dataset/dogbreeds/'
labels_file = pd.read_csv(f'{PATH}labels.csv')
dataset = labels_file
print(dataset.head())
dataset['path'] = [f'{PATH}train/{x}.jpg' for x in dataset['id']]
print(dataset.shape)
print(dataset.head())

# dataset3 = dataset.loc[]

# breed_names = dataset['breed'].unique()
#
#
# # dict1 = {y:x for y,x in enumerate(tmp)}
# dict2 = {x: y for y, x in enumerate(breed_names)}
#
# dataset['bin_breeds'] = [dict2[x] for x in dataset['breed']]
#
# def get_vector(idx):
#     zeroV = np.zeros((120,), dtype=int)
#     zeroV[idx['bin_breeds']] = 1
#     return zeroV
#
# dataset['vector'] = dataset.apply(get_vector, axis=1)
# index = 0
# full_list = []
# X_train = []
# y_train = []
# X_test = []
# y_test = []
#
# for idx, row in dataset.iterrows():
#     try:
#         image = cv2.imread(row['path'])#,cv2.IMREAD_GRAYSCALE)
#         # grey = cv2.cvtColor(image)#, cv2.COLOR_RGB2GRAY)
#     except Exception as e:
#         print(f'Error at file index {index} with path:', row['path'], sep='')
#         pass
#
#     image_resize = cv2.resize(image, (resolution, resolution))
#     full_list.append([image_resize, row['vector']])
#     index += 1
#     if index > input_size: break
#     if index%500==0:print(index)
#
# random.shuffle(full_list)
# index = 0
# if test_data == True:
#     for x, y in full_list:
#         index += 1
#         if index % 10 != 0:
#             X_train.append(x)
#             y_train.append(y)
#         else:  # every 10th sample is a test sample
#             X_test.append(x)
#             y_test.append(y)
#
#     # print(f'Overall table size:', len(y_test))
#     print('Please remember to normalize and reshape || simple_norm and simple_reshape')
#
#     return (X_train, np.array(y_train), X_test, np.array(y_test))
#
# else:
#
#     for x, y in full_list:
#         X_train.append(x)
#         y_train.append(y)
#     # print(f'Overall table size:', len(y_test))
#     print('Please remember to normalize and reshape || simple_norm and simple_reshape')
#
#     return (X_train, np.array(y_train))