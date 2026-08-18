# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:03:11 2026

@author: amana
"""

#Step 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

#Step-> It's time to import the dataset
dataset = pd.read_csv(
    r"C:\Data_Science\Machine_Learning_Model_Let's Goo\Non_Linear_Regression\Polynomial_Regression\Dataset\Position_Salaries.csv"
)

df=pd.DataFrame(dataset)
print(df)


#Step-2 TO identify the X and Y
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Step-3 To visullize the data

plt.scatter(X, y)
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Level vs Salary")
plt.show()

#Step-4 Fit Linear Regression First

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Step-5 Genuiely what is the problem from the Linear Regression Let's see
#I want to check the how much salary i have to give the employee whose level ins 6.5
lin_reg.predict([[6.5]])

#Step 6 — Create Polynomial Features
#Level ka square because we take the degree 2 if degree 3 then take the cube of the level 
from sklearn.preprocessing import PolynomialFeatures

#It's overfitting at degree =2
#poly_reg = PolynomialFeatures(degree=2)

#Perfect at the degreee=5
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)


#Train the Polynomial Regression Model
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

#Step 8 — Predict Level 6.5 using Polynomial Regression

lin_reg_poly.predict(poly_reg.transform([[6.5]]))

#Step 9 — Visualize Polynomial Regression
plt.scatter(X, y)
plt.plot(X, lin_reg_poly.predict(X_poly))
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.show()