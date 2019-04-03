from models import manipulation, kerasmodels
from data import dogcat
from tasking import general

resolution = 50
epoch = 10
input_size = 3000
shape = (resolution,resolution,1)

X_train,y_train,X_test,y_test = dogcat.load_data(input_size=input_size,resolution=resolution,test_data=True)
X_train = general.simple_reshape(X_train)
X_train = general.simple_norm(X_train)

# model_trained = kerasmodels.modelSeq1(shape,classes=1)
# model_trained.fit(X_train,y_train,batch_size=50,epochs=epoch,validation_split=0.2 )
#
model_lenet = kerasmodels.modelLeNet(shape,classes=1)
model_lenet.fit(X_train,y_train,batch_size=50,epochs=epoch,validation_split=0.2)


#1 simple = loss: 0.3291 - acc: 0.8565
#2 simple = loss: 0.1541 - acc: 0.9460

#1 lenet =
#2 lenet =