import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Read in the data file
# More information on the dataset can be found here: https://www.kaggle.com/datasets/vipullrathod/fish-market
input_data = pd.read_csv('coefficient-of-determination\Fish-Market.csv')

# print(input_data.head())

# I want to include the 'Species' column with our linear regression.
# It is a categorical column, so it will need to be encoded into dummy variables
species_values = input_data['Species'].unique()

# print(species_values)

# We define the mapping based on the data
category_to_numeric_mapping = { 'Bream' : 0, 'Roach' : 1, 'Whitefish' : 2, 'Parkki' : 3, 'Perch' : 4, 'Pike' : 5, 'Smelt': 6 }
numeric_to_category_mapping = { 0 : 'Bream', 1 : ' Roach', 2 : 'Whitefish', 3 : 'Parkki', 4 : 'Perch', 5 : 'Pike', 6 : 'Smelt'}

input_data['Species_encoded'] = input_data['Species'].map(category_to_numeric_mapping)

# print(input_data.head())

# For this exercise I want to predict the weight based on the other characteristics
X_columns = ['Species_encoded', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
y_column = 'Weight'

X = input_data[X_columns]
y = input_data[y_column]

# We split the X and y data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

# establish and "train" the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_test)

# coefficient of determination is "r2_score" or can be
determination_coefficient = r2_score(y_test, predictions)

# here we print the coefficient
print("coefficient: ", determination_coefficient)


# write it out to a file
# with open('coefficient_of_determination.txt', 'w') as file:
#  file.write("Coefficient of Determination is: ", determination_coefficient
