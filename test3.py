from models import kerasmodels
from data import dogcat
from tasking import general
from tensorflow.contrib.keras import models
import numpy as np
import pandas as pd

display = 5

X_test, y_test, filenames = dogcat.load_general_patch(resolution=100,path='C:/Dataset/img/Test',input_size=10)
# general.plots(X_test[:5],titles=y_test[:5])

X_test_reshape = general.simple_reshape(X_test,100)
X_test_reshape = general.simple_norm(X_test_reshape)

model = models.load_model('repo/DogCat100.h5')
predictions = model.predict(X_test_reshape)

# print(test_data.shape)
predictions = np.array(predictions).reshape(len(predictions), )
print(filenames)
print(y_test)

ds = pd.DataFrame()
ds['index'] = range(len(y_test))
ds['filename'] = pd.Series(filenames)
ds['y_test'] = pd.Series(y_test)
ds['predictions'] = pd.Series(predictions)
print(ds.dtypes)
# print(ds.iloc[:display,:])
# general.plots(X_test, titles=predictions,rows=5)

# print(ds)
##############################################
#few most correct dogs
# most_correct_dogs = dogcat.most_correct_dogs(ds)
# tmp_list = []
# for x in list(most_correct_dogs['index']):
#     tmp_list.append(X_test[x])

# print(most_correct_dogs)
# general.plots(tmp_list[:display], titles=list(most_correct_dogs['predictions']))
######################################################################################
#few most correct cats
# ds.sort_values[ds['predictions']](by=['Brand'], inplace=True, ascending=False)

#just correct no sorting
# ds.sort_values[ds['predictions']](by=['Brand'], inplace=True, ascending=False)

#corrext size of Dog/Dog table print
# round prediction of Keras
# A few correct labels at random
# A few incorrect labels at random
# The most correct labels of each class (i.e. those with highest probability that are correct)
# The most incorrect labels of each class (i.e. those with highest probability that are incorrect)
# The most uncertain labels (i.e. those with probability closest to 0.5).
#