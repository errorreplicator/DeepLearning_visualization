import pandas as pd
import numpy as np

# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)
# pd.set_option('display.width', 1000)
# pd.set_option('display.large_repr', 'truncate')
# pd.set_option('display.max_columns', 0)

pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_seq_items',None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)
############# 4 columns width #################
pd.set_option('display.width', 1000)
pd.set_option('display.max_column',None)
pd.set_option('display.max_colwidth', 500)
###############################################
# generate random dataset
ds = pd.DataFrame(np.random.choice([0.9,4.1,'2ta',np.nan], size=(10,5)),columns=['fir','sec','thir','4th','5th'])
ds =  np.zeros((3,3))
ds = np.ones((3,3))
ds =  np.eye(3)
# renaming columns
ds.rename(columns={'col1':'First','col2':'2nd','col3':'3rd','Unnamed: 4':'4th'}, inplace=True)

# dropping column
ds.drop('Unnamed: 0',1,inplace=True)

# set value

full.set_value([61,829],'Embarked','S')

# -------------------------------RANGE -----------------------------------------------------

     val1  val2 val3
n
100    10    10    C
101    10    20    C
102    10    30    B
103    10    40    B
104    10    50    A

In: dataset['val3'][104]
Out: 'A'
In: dataset.loc[104, 'val3']
Out: 'A'
In: dataset.ix[104, 'val3']
Out: 'A'
In: dataset.ix[104, 2]
Out: 'A'
In: dataset.iloc[4, 2]
Out: 'A'


In: dataset[['val3', 'val2']][0:2]
In: dataset.loc[range(100, 102), ['val3', 'val2']]
In: dataset.ix[range(100, 102), ['val3', 'val2']]
In: dataset.ix[range(100, 102), [2, 1]]
In: dataset.iloc[range(2), [2,1]]
Out:
  val3  val2
n
100  C  10
101  C  20

#print row 1to 4  of columns starting from name '2nd' to end of column list
print(ds.loc[1:4,'2nd':])

# reads csv and providing names of the columns
ds = pd.read_csv('census.dat',names=['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification'])

#read csv ignoring spaces in columns
ds = pd.read_csv('census.dat', skipinitialspace=True)

#stripe out spaces in the x2 column
ds['x2'] = pd.core.strings.str_strip(ds['x2'])

# ??????????
# print(ds['capital-gain'][ds['capital-gain'] == '?'])
# print football[(football.wins > 10) and (football.team == "Packers")]


#prints out 5th column for only the rows where in fir = 0.9
print(ds['5th'][ds['fir']  == 0.9])
#converts column fir to number
ds.fir = pd.to_numeric(ds.fir)
##ds['rc'] = ds['rc'].astype(np.float64)
##ds['wc'] = ds['wc'].astype(np.int64)

# print types of columns
print(ds.dtypes)


# creating column base on values in different column
#simple example
ds['Gender'] = ds['Sex'].map({'male': 0, 'female':1}).astype(np.int)
#more advance example
def realF (a):
    if (a['Sex'] == 'female' or a['Pclass'] == 3 or a['Age'] < 18):
        return 1
    else:
        return 0

ds_titan['SurvivalRate'] = ds_titan.apply(realF,axis=1)

#group by
for i in range(1,4):
    print (i, len(ds_train[ (ds_train['Sex'] == 'male') & (ds_train['Pclass'] == i) ]))

# simple mean of one column
print(ds_train['Survived'].mean())
# advanced filtering
print(ds_train[(ds_train['Survived'] == 1) & ((ds_train['Sex'] != 'male') | (ds_train['Sex'] == 'male' ))])
print(ds_train.loc[ds_train['Age'].isnull(),['Name','Sex','Age']])

# Display only objects
print(ds.dtypes[ds.dtypes.map(lambda x: x=='object')])

# Count the number of NaNs each column has.
print("\n*** NaNs per column:")
print(pd.isnull(train).sum())

#List all values in column and cound occurences

print(pd.value_counts(full["CabinType"]))

# ----------------Filtering and applying the data -------------------

def change_zero_age(column): #chaning all passed values to 1
    return 1

train.loc[(train['Age']<0.999),'Age'] = train.loc[(train['Age']<0.999),'Age'].apply(change_zero_age) #<-- error working on a copy

train['Age'][train['Age']<0.999] = train['Age'][train['Age']<0.999].apply(change_zero_age) # <-- correct

# ----------------------------CATEGORICAL VALUES ------------------------
full['EmbarkedT'] = pd.Categorical(full['Embarked'], categories=[]).codes #914 2
full2 = pd.get_dummies(full['Embarked'])

#reset index
full.reset_index(inplace=True) ############# REINDEX

#concatenate
full = pd.concat([train,test],axis=0)

subsetDataFrame = dfObj[dfObj['Product'].isin(['Mangos', 'Grapes'])] # in SQL