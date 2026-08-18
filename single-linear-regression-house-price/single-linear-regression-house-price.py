# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:41:44 2026

@author: amana
"""

# import the dataset
import pandas as pd
import numpy as np

# Let's do the Phase-1 Task
 # Total rows kitni hain?
 # Total columns kitne hain?
 # Target (Dependent Variable) kaunsi hai?
 # Independent Variable kaunsi loge?
 # Kya missing values hain?
 # Kya categorical values hain?
 
 # Let's import the dataset
 dataset=pd.read_csv("C:\Data_Science\Machine_Learning_Model_Let's Goo\Single_Linear_Regression_Model\single_linear_regression_house_prices.csv")
 
 #Let's to convert in to the table format 
df=pd.DataFrame(dataset)
print(df)


dataset.shape[0] 
dataset.shape[1]

Y = dataset.iloc[:, -1].values  # dependent variable
print(Y)

X = dataset.iloc[:, :-1].values   #Independent variables 
print(X)

# Kis column mein kitne missing values hain
dataset.isnull().sum()

# Missing values wale column names
dataset.columns[dataset.isnull().sum() > 0]

dataset.info()

# Let's do the Phase-2 Task->Model Building
 # LinearRegression import
 # Object create
 # fit() run
 
 from sklearn.linear_model import LinearRegression
 model = LinearRegression()
 
 model.fit(X, Y)
 
 #Now to predict the model by using the dependent varibale and see the single linear regression
 
 # predicted_price = model.predict([[2000]])
 # print(predicted_price)``````
 
 # predicted_price = model.predict([[1500]])
 # print(predicted_price)
 
  predicted_price= model.predict([[300]])
  print(predicted_price)
 
    
 #Phase #3 Visullization 
# %%
 # Plot the original data points.

 import matplotlib.pyplot as plt


plt.scatter(X, Y)

plt.plot(X, model.predict(X), color='red')

plt.title("Simple Linear Regression")
plt.xlabel("Area")
plt.ylabel("Price")

plt.show()


print(model.coef_)
 
 
 

 
 
 
 

 