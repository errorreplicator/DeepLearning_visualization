from models import kerasmodels
from data import dogcat
from tasking import general
resolution = 100
epoch = 10
shape = (resolution,resolution,1)


model = kerasmodels.modelSeq1(shape,1)
X_train,y_train = dogcat.load_data(resolution=100,test_data=False)
X_train = general.simple_reshape(X_train,100)
X_train = general.simple_norm(X_train)

model.fit(X_train,y_train,batch_size=50,epochs=epoch)

model.save('repo/DogCat100.h5')

# load test data for prediction [image,real_class,filename]
# Predict classes of an image
# save prediction as [prediction,real class,filename]
# write function to print
# A few correct labels at random
# A few incorrect labels at random
# The most correct labels of each class (i.e. those with highest probability that are correct)
# The most incorrect labels of each class (i.e. those with highest probability that are incorrect)
# The most uncertain labels (i.e. those with probability closest to 0.5).
#