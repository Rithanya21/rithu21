# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:18:56 2020

@author: rithanya.s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
data
data.head()
X = data.drop("Scores",axis=1).values
y = data['Scores'].values 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
plt.scatter(X_train, y_train)
plt.plot(X_train, 1.9322+9.94167834*X_train, 'black')
plt.show()
print("Training set score:",lr.score(X_train, y_train))
print("Test set score:",lr.score(X_test, y_test))
print("Coeff:     ",lr.coef_)     
print("Intercept: ",lr.intercept_)
Hours = float(input("Enter Study Hours: "))
Hours = [[Hours]]
lr.predict(Hours)


