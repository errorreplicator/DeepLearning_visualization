# from tensorflow.contrib import keras
from  models import kerasmodels

model = kerasmodels.modelSeq1((50,50,1))

print(model.summary())