# --------------
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

## Load the data
data = pd.read_csv(path)
print(data.shape)
print(data.columns)
## Split the data and preprocess
print(data['source'].value_counts())

train = data[data['source'] == 'train']
test = data[data['source'] == 'test']

## Baseline regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_base = train[['Item_Weight','Item_MRP','Item_Visibility']]
X_base_train, X_base_val, y_train, y_val = train_test_split(X_base,train.Item_Outlet_Sales, test_size=0.30,random_state=42)
print(X_base_train.shape,X_base_val.shape, y_train.shape,y_val.shape)

## Effect on R-square if you increase the number of predictors
Model_b1 = LinearRegression(normalize=True)
Model_b1.fit(X_base_train, y_train)
pred_b1 = Model_b1.predict(X_base_val)

MSE_bl = np.mean(pred_b1 - y_val)**2

#R-Square
R2_bl = r2_score(y_val, pred_b1)

print('Baseline model', MSE_bl, R2_bl)
## Effect of decreasing feature from the previous model
X_2 = train.drop(columns = ['Item_Outlet_Sales','Item_Identifier', 'source'])
X2_base_train, X2_base_val, y2_train, y2_val = train_test_split(X_2,train.Item_Outlet_Sales, test_size=0.30,random_state=42)

Model_2 = LinearRegression(normalize=True)
Model_2.fit(X2_base_train, y2_train)
pred_2 = Model_2.predict(X2_base_val)

MSE_2 = np.mean(pred_2 - y2_val)**2
#R-Square
R2_2 = r2_score(y2_val,pred_2)
print('Second model', MSE_2, R2_2)

#Third model
X_3 = train.drop(columns = ['Item_Outlet_Sales','Item_Identifier', 'Item_Visibility', 'Outlet_Years', 'source'])
X3_base_train, X3_base_val, y3_train, y3_val = train_test_split(X_3,train.Item_Outlet_Sales, test_size=0.30,random_state=42)

Model_3 = LinearRegression(normalize=True)
Model_3.fit(X3_base_train, y3_train)
pred_3 = Model_3.predict(X3_base_val)

MSE_3 = np.mean(pred_3 - y3_val)**2
#R-Square
R2_3 = r2_score(y3_val,pred_3)
print('Third model', MSE_3, R2_3)
## Detecting hetroskedacity


## Model coefficients


## Ridge regression


## Lasso regression


## Cross vallidation



