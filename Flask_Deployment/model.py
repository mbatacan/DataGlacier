import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


df = pd.read_csv('Toy_dataset.csv')
print(df.columns)
print(df.shape)
print(df.dtypes)

y = df.Income
X = df.drop(['Income'],axis = 1)

print(X['City'].value_counts(sort = True))

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


#Split outcomes
X_train_full, X_valid_full , y_train, y_valid = train_test_split(X,y,train_size = 0.8,test_size = 0.2,random_state = 0)


#Preprocessing Data
null_cols = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
X_train_full.drop(null_cols,axis = 1, inplace = True)
X_valid_full.drop(null_cols,axis= 1, inplace = True)

X_train_full.drop(columns = ['Number'], axis = 1)
X_valid_full.drop(columns = ['Number'], axis = 1)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64','float64']]


print('Low Cardinality cols: ')
print(low_cardinality_cols)
print('Numerical Cols: ')
numerical_cols.remove('Number')
print(numerical_cols)

my_cols = low_cardinality_cols + numerical_cols

X_train = X_train_full[my_cols]
X_valid = X_valid_full[my_cols]

print('Valid Columns:')
print(X_train.columns)

object_cols = [col for col in X_train if X_train[col].dtype == 'object']

OH_encoder = OneHotEncoder(handle_unknown = 'ignore',sparse = False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

num_X_train = X_train.drop(object_cols,axis = 1)
num_X_valid = X_valid.drop(object_cols,axis = 1)

OH_X_train = pd.concat([num_X_train,OH_cols_train],axis = 1)
OH_X_valid = pd.concat([num_X_valid,OH_cols_valid],axis = 1)


regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(OH_X_train,y_train)
preds = regressor.predict(OH_X_valid)
print(preds[:5])

pickle.dump(regressor,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


