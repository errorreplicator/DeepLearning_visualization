from models import kerasmodels
from data import dogcat
from tasking import general
from tensorflow.contrib.keras import models
import numpy as np
import pandas as pd

display = 5

X_test, y_test, filenames = dogcat.load_general_patch(resolution=100,path='C:/Dataset/img/Test',input_size=10000)

X_test_reshape = general.simple_reshape(X_test,100)
X_test_reshape = general.simple_norm(X_test_reshape)

model = models.load_model('repo/DogCat100.h5')
predictions = model.predict(X_test_reshape)

predictions = np.array(predictions).reshape(len(predictions), )
predictions = [round(a,4) for a in predictions] # round prediction to 5 digits / no need more


ds = pd.DataFrame()
ds['index'] = range(len(y_test))
ds['filename'] = pd.Series(filenames)
ds['y_test'] = pd.Series(y_test)
ds['predictions'] = pd.Series(predictions)

######################################################################################
#few most correct dogs
indexes,predict = dogcat.most_correct_dogs(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most correct dogs')
######################################################################################
#few most incorrect dogs
indexes,predict = dogcat.most_incorrect_dogs(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most incorrect dogs')
######################################################################################
#few most uncertain dogs
indexes,predict = dogcat.most_uncertain_dogs(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most uncertain dogs')
#just correct no sorting
######################################################################################

######################################################################################
#few most correct cats
indexes,predict = dogcat.most_correct_cats(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most correct cats')
######################################################################################
#few most incorrect cats
indexes,predict = dogcat.most_incorrect_cats(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most incorrect cats')
######################################################################################
#few most uncertain cats
indexes,predict = dogcat.most_uncertain_cats(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most uncertain cats')
#just correct no sorting
######################################################################################


#corrext size of Dog/Dog table print
# round prediction of Keras
# A few correct labels at random
# A few incorrect labels at random
# The most correct labels of each class (i.e. those with highest probability that are correct)
# The most incorrect labels of each class (i.e. those with highest probability that are incorrect)
# The most uncertain labels (i.e. those with probability closest to 0.5).
#