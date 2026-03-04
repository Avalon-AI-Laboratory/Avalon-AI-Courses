import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("adult.csv")

# Filling missing values
df.replace("?", np.nan, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Discretization (simplifying marital status)
df.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
            'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married',
            'not married', 'not married', 'not married'], inplace=True)

# Label Encoding
category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                'relationship', 'gender', 'native-country', 'income']
label_encoder = preprocessing.LabelEncoder()

# Creating a mapping dictionary
mapping_dict = {}
for col in category_col:
    df[col] = label_encoder.fit_transform(df[col])
    mapping_dict[col] = dict(enumerate(label_encoder.classes_))

print(mapping_dict)

# Dropping redundant columns
df.drop(['fnlwgt', 'educational-num'], axis=1, inplace=True)

# Splitting features and target
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Initialize and Train Decision Tree Classifier
dt_clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)

# Predictions
y_pred_gini = dt_clf_gini.predict(X_test)

# Accuracy Score
print("Decision Tree using Gini Index\nAccuracy:", accuracy_score(y_test, y_pred_gini) * 100)

# Save Model Using Pickle
with open("model.pkl", "wb") as model_file:
    pickle.dump(dt_clf_gini, model_file)
    pickle.dump(mapping_dict, open("mapping.pkl", "wb"))