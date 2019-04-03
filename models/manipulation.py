import time
import datetime
from tensorflow.contrib.keras import models

def retTime():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    return st

def saveModelAll(model,filename):
    model.save(f'repo/{filename}.{retTime()}.h5')

def loadFileModel(path):
    model = models.load_model(path)
    return model

def loadWeights(model,path):
    model = model.load_weights(path)
    return model

def saveWeights(model,filename):
    model.save_weights(f'repo/{filename}.h5')

def modelSummary(model,summary=True,weights=True,optimizer=True):

    if weights:
        print(model.get_weights())
    if summary:
        print(model.summary())
    if optimizer:
        print(model.optimizer)