



















#-------------------INPUT SHAPE -------------------------------------#

# from tensorflow.contrib.keras import models,layers
# from tensorflow.contrib.keras import callbacks
# from time import time
# X_train = [0,1,2,3,4]
# y_train = [0,2,4,6,8]
# tensorboard = callbacks.TensorBoard(log_dir=f'C:/log/test.log')
#
# model = models.Sequential()
#
# model.add(layers.Dense(3,activation='relu',input_shape=(1,)))
# # model.add(layers.Dense(3,activation='linear'))
# model.add(layers.Dense(1,activation='linear'))


# print(model.summary())
# model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'],)
# model.fit(X_train,y_train,epochs=5)#,callbacks=[tensorboard])


# predictivemodel = models.Sequential()
# predictivemodel.add(layers.Dense(32, input_shape=(2,2,7)))#, W_regularizer=WeightRegularizer(l1=0.000001,l2=0.000001), init='normal'))
# predictivemodel.add(layers.Dense(8))#, W_regularizer=WeightRegularizer(l1=0.000001,l2=0.000001), init='normal'))
# predictivemodel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#
# 514*514+514= 264 710
# 514*257+257= 132 355
#
# print(predictivemodel.summary())