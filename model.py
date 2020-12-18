# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:18:08 2020

@author: Nagesh
"""

#%%


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#%%

data = pd.read_excel(open("dataset.xlsx", "rb"), sheet_name = "Sheet1")
print(data.head())

#%%

x, y = data.iloc[:, :-1], data.iloc[:, -1]
print(x.shape, y.shape)

#%%

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#%%

model = LinearRegression()
model.fit(x_train, y_train)

#%%

print("model score = ", model.score(x_test, y_test))
print("model_mean_absolute_error = ", mean_absolute_error(y_test, model.predict(x_test)))


#%%

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

