from models import kerasmodels
from data import dogcat
import numpy as np
import keras
resolution = 50
epoch = 10
input_size = 3000

X_train,y_train = dogcat.load_data(input_size,resolution)

X = np.array(X_train).reshape(-1,resolution,resolution,1)
print(X.shape[1:])
model = kerasmodels.modelSeq1(X.shape[1:])
X_simple_div=X/255.0
X = keras.utils.normalize(X,axis=1)
np.set_printoptions(threshold=np.inf)

model.fit(X_simple_div,y_train,validation_split=0.2 ,epochs=epoch,batch_size=50,verbose=2)

#1 /255 loss: 0.2324 - acc: 0.9078
#2 /255 loss: 0.3290 - acc: 0.8525
#3 /255 loss: 0.3129 - acc: 0.8640

#1 keras.utils.normalize loss: 0.4403 - acc: 0.7928
#2 loss: 0.3445 - acc: 0.8468
#3 loss: 0.4183 - acc: 0.8058