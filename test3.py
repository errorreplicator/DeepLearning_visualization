from models import kerasmodels
from data import dogcat
from tasking import general
from tensorflow.contrib.keras import models
import numpy as np
import pandas as pd

display = 5

X_test, y_test, filenames = dogcat.load_general_patch(resolution=100,path='C:/Dataset/img/Test',input_size=1000)
# general.plots(X_test[:5],titles=y_test[:5])

X_test_reshape = general.simple_reshape(X_test,100)
X_test_reshape = general.simple_norm(X_test_reshape)

model = models.load_model('repo/DogCat100.h5')
predictions = model.predict(X_test_reshape)

# print(test_data.shape)
predictions = np.array(predictions).reshape(len(predictions), )
print(filenames[:display])
print(y_test[:display])

ds = pd.DataFrame()
ds['filename'] = pd.Series(filenames)
ds['y_test'] = pd.Series(y_test)
ds['predictions'] = pd.Series(predictions)
print(ds.iloc[:display,:])
general.plots(X_test[:display], titles=predictions[:display])

#corrext size of Dog/Dog table print
# add index to dataframe
# A few correct labels at random
# A few incorrect labels at random
# The most correct labels of each class (i.e. those with highest probability that are correct)
# The most incorrect labels of each class (i.e. those with highest probability that are incorrect)
# The most uncertain labels (i.e. those with probability closest to 0.5).
#