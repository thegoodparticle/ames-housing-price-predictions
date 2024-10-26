# Demonstrate One-hot encoding and Label encoding in Python
# 1. Importing the Libraries
import pandas as pd
import numpy as np

# 2. Reading the file
df = pd.read_csv("../dataset/AmesHousingDataset.csv")
print(df.head())

for column in df.columns:
    if df[column].dtype in [np.int64, np.float64]:  # Numeric columns
        df[column].fillna(df[column].mean(), inplace=True)
    elif df[column].dtype == "object":  # Categorical columns
        df[column].fillna(df[column].mode()[0], inplace=True)

        dummies_column = pd.get_dummies(df[column], prefix=column, dtype=int)
        df = pd.concat([df, dummies_column], axis=1)
        df = df.drop(column, axis=1)

print(df.head())

# print(df['Lot Config'].value_counts())
# one_hot_encoded_data = pd.get_dummies(df, columns = ['Utilities', 'Lot Config'], dtype=int)
# print(one_hot_encoded_data.head())

# One hot encoding for Covid_Severity field
# print(df['Geography'].value_counts())
# one_hot_encoded_data = pd.get_dummies(df, columns = ['Geography', 'Gender'], dtype=int)
# print(one_hot_encoded_data)

# For multiple columns
# one_hot_encoded_data = pd.get_dummies(df, columns = ['Covid_Severity', 'Gender'])
# print(one_hot_encoded_data)
