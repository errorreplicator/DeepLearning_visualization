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

model_trained = kerasmodels.modelSeq1(shape)
model_trained.fit(X_train,y_train,batch_size=50,epochs=epoch,validation_split=0.2 )

model_raw = kerasmodels.modelSeq1(shape)
model_raw.load_weights('repo/CatDog1_w.h5')

model_saved = manipulation.loadFileModel('repo/CatDog1.20190330194006.h5')
X_test, y_test = dogcat.load_TestData(50)
X_test = general.simple_reshape(X_test)
X_test = general.simple_norm(X_test)

predictions_m_trained = model_trained.predict(X_test)
prediction_m_raw = model_raw.predict(X_test)
predictions_m_saved = model_saved.predict(X_test)

print(f'Number of test samples:{len(y_test)}',sep='\n')

print('predictions_m_trained below', sep='\n')
for x in predictions_m_trained:
    print(x)
print('')

print('predictions_m_raw below',sep='\n')
for y in prediction_m_raw:
    print(y)
print('')

print('predictions_m_saved below',sep='\n')
for z in predictions_m_saved:
    print(z)
print('')

print(y_test)



