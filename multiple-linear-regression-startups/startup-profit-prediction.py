# Step 1 : Understand the Business Problem

# Step 2 : Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 3 : Import the Dataset
dataset = pd.read_csv(
    r"C:\Data_Science\Machine_Learning_Model_Let's Goo\Multiple_Linear_Regression_Mode\50_Startups_dataset.csv",index_col=0
)

# Step 4 : Understand the Dataset
print(dataset.head())
print(dataset.tail())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Check Missing Values
print(dataset.isnull().sum())

# Check Duplicate Values
print(dataset.duplicated().sum())

# Step 5 : Identify X and Y
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Step 6 : Check Missing Values
# No missing values -> Skip

# Step 7 : Encode Categorical Data
# Step 7 : Convert Categorical Values into One-Hot Encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(sparse_output=False, drop='first'), [3])],
    remainder='passthrough'
)

X = ct.fit_transform(X)
# Step 8 : Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=0
)

# Step 9 : Train the Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, Y_train)

#Predict the profit dude
Y_pred = model.predict(X_test)

print(Y_pred)

print("Actual Profit:")
print(Y_test)

print("Predicted Profit:")
print(Y_pred)


#Just to check how much my model accuracy is perfect as compare to our testing data Let's goo 
from sklearn.metrics import r2_score

r2 = r2_score(Y_test, Y_pred)

print("R² Score:", r2)



#Let's do our Model for the Businness Perpective
#We get here the 5 values jaha pr  3 is our state-new work,floridia and  but Californiais our base line means ? 
#California ka estimated effect California ke comparison mein -959 hai.
#New York ka estimated effect California ke comparison mein +699 hai.

#   When we need to know about which  categoires values is our base and base categoires values is not going to effect on the profit 
print(model.coef_)
print(ct.named_transformers_['encoder'].categories_) 
print(X.shape);

print(ct.named_transformers_['encoder'].categories_)

print(ct.get_feature_names_out())

#Business man told you hey i have only 3 lakh bhai where i need to invest to get the max profit so we can ealsiy say them 

#Step-1 Make Scenrio
scenario_data = np.array([
    [200000, 50000, 50000, "California"],
    [50000, 50000, 200000, "California"],
    [100000, 100000, 100000, "California"]
], dtype=object)


#How much money basically i inverst according to the cloumn but state always neeed to take base value california 
scenario_X = ct.transform(scenario_data)

scenario_profit = model.predict(scenario_X)

print(scenario_profit)


print("R&D Focused Profit:", scenario_profit[0])
print("Marketing Focused Profit:", scenario_profit[1])
print("Balanced Profit:", scenario_profit[2])