from models import kerasmodels
from data import dogcat
from tasking import general

resolution = 50
epoch = 1
input_size = 300

X_train,y_train,x_test,y_test = dogcat.load_data(input_size,resolution)

X_train = general.simple_reshape(X_train)
x_test = general.simple_reshape(X_train)

X_train = general.simple_norm(X_train)
x_test = general.simple_reshape(x_test)


model = kerasmodels.modelSeq1(X_train.shape[1:])


model.fit(X_train,y_train,validation_split=0.2 ,epochs=epoch,batch_size=50)
kerasmodels.saveModelAll(model,'testowy1')

#1 /255 loss: 0.2324 - acc: 0.9078
#2 /255 loss: 0.3290 - acc: 0.8525
#3 /255 loss: 0.3129 - acc: 0.8640

#1 keras.utils.normalize loss: 0.4403 - acc: 0.7928
#2 loss: 0.3445 - acc: 0.8468
#3 loss: 0.4183 - acc: 0.8058