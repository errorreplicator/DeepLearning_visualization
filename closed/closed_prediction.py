from models import manipulation, kerasmodels
from data import dataload
from tasking import general

resolution = 50
epoch = 10
input_size = 3000
shape = (resolution,resolution,1)
X_train,y_train,X_test,y_test = dataload.load_data(input_size=input_size, resolution=resolution, test_data=True)
X_train = general.simple_reshape(X_train)
X_train = general.simple_norm(X_train)

model_trained = kerasmodels.modelSeq1(shape)
model_trained.fit(X_train,y_train,batch_size=50,epochs=epoch,validation_split=0.2 )

model_raw = kerasmodels.modelSeq1(shape)
model_raw.load_weights('repo/CatDog1_w.h5')

model_saved = manipulation.loadFileModel('repo/CatDog1.20190330194006.h5')

X_test, y_test = dataload.load_TestData(50)
X_test = general.simple_reshape(X_test)
X_test = general.simple_norm(X_test)

predictions_m_trained = model_trained.predict(X_test)
prediction_m_raw = model_raw.predict(X_test)
predictions_m_saved = model_saved.predict(X_test)

predictions_trained_bin = model_trained.predict_classes(X_test)
prediction_raw_bin = model_raw.predict_classes(X_test)
predictions_saved_bin = model_saved.predict_classes(X_test)

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

print('predictions_trained_bin below', sep='\n')
for x in predictions_trained_bin:
    print(x)
print('')

print('predictions_raw_bin below',sep='\n')
for y in prediction_raw_bin:
    print(y)
print('')

print('predictions_saved_bin below',sep='\n')
for z in predictions_saved_bin:
    print(z)
print('')


print(y_test)

##############################PLOTTING THE PREDICTION DATA####################################################

X_test, y_test, filenames = dataload.load_general_patch(resolution=100, path='C:/Dataset/img/Test', input_size=10000)

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
indexes,predict = dataload.most_correct_dogs(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most correct dogs')
######################################################################################
#few most incorrect dogs
indexes,predict = dataload.most_incorrect_dogs(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most incorrect dogs')
######################################################################################
#few most uncertain dogs
indexes,predict = dataload.most_uncertain_dogs(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most uncertain dogs')
#just correct no sorting
######################################################################################

######################################################################################
#few most correct cats
indexes,predict = dataload.most_correct_cats(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most correct cats')
######################################################################################
#few most incorrect cats
indexes,predict = dataload.most_incorrect_cats(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most incorrect cats')
######################################################################################
#few most uncertain cats
indexes,predict = dataload.most_uncertain_cats(ds)
general.plots([X_test[x] for x in indexes][:display],titles=predict[:display],plot_title='most uncertain cats')
#just correct no sorting
######################################################################################