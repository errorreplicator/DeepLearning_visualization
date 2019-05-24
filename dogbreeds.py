import pandas as pd
from data import dataload
from tasking import general
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

X_train, y_train, X_test, y_test = dataload.dogbreeds(100,resolution=64,test_data=True)

X_train = general.simple_reshape(X_train,64)
X_test = general.simple_reshape(X_test,64)
X_train = general.simple_norm(X_train)
X_test = general.simple_norm(X_test)


print(y_train)

