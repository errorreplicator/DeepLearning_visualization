from models import kerasmodels
import h5py

model = kerasmodels.loadModel('repo/testowy1.20190325205434.h5')
print(model.summary())