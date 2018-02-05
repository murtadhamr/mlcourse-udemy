# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:57:32 2018

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fitting RF regression model to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

#predicting y from fitted model
y_pred = regressor.predict(6.5)

#visualizing RF regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth of Bluff (RF Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
