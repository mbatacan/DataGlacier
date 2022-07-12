import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('Toy_dataset.csv')
print(df.columns)
print(df.shape)



from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
y = df.Income
X = df.select_dtypes(exclude =['object'])
X = X.drop(['Income'],axis = 1)
regressor.fit(X,y)
print(X.head())

pickle.dump(regressor,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[3,35]]))








