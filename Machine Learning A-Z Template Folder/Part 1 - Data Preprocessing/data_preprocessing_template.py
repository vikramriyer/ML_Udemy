# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# the part to the left of comma is for rows, i.e. consider all the rows => independent variables
# the part to the right of comma is for columns, i.e. take all but the last one, as -1 acc to a list is last => dependent variable
X = dataset.iloc[:, :-1].values

# same for this one, where we say all the rows be taken, and only the third column be taken
y = dataset.iloc[:, 3].values

# Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Categorical data
from sklearn.preprocessing import LabelEncoder
labelencode_X = LabelEncoder()

# what we excatly want to do is, encode the categorical columns into numbers so that, they are easy to use
# machine learning uses equations to do the calculations, so using numbers seems obvious, => very imp 
X[:, 0] = labelencode_X.fit_transform(X[:, 0])

# let see the problems above method created
# 1. now, if we print and see the values for country column, we can see values as 0(Spain), 1(France), 2(Germany).
# now the machine learning models may think that, Spain < France < Germany and introduce a problem, but these
# countries do not follow any such order, they are just different countries
# 2. We also lost some data, so, we need to first store that 0 - spain, 1 - france and 2 - germany and then
# do the transformations

# lets look at a more robust way of doing things which makes the data more clear for the machine learning models
# we will create new columns for each value in the category, i.e. the dummy variables, so for ex: instead of
# country column, we will have france, germany, spain and use 0 and 1 values to say if that particular row 
# has that country
# for ex:
# ----------                                 -----------------------------------
# |country |                                 | France    | Germany   | Spain   |
# ---------|                                 -----------------------------------
# |France  |   ====> is transformed to ====> | 1         | 0         | 0       |
# |Germany |                                 | 0         | 1         | 0       |
# |Spain   |                                 | 0         | 0         | 1       |
# ----------                                 -----------------------------------

from sklearn.preprocessing import OneHotEncoder
onehotencode = OneHotEncoder()
X = onehotencode.fit_transform(X).toarray()

print X

# lets do the encoding for purchases column
labelencode_y = LabelEncoder()
y = labelencode_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""