# To run the file in terminal type > python workflow.py 
# Has to connect with Prefect cloud -> https://app.prefect.cloud/
# Prefect Login [Prefect Cloud] - https://www.prefect.io/opensource  -> get started

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from prefect import flow, task

# Step 2: Load the Dataset
@task
def load_dataset():
    return pd.read_csv("../../dataset/AmesHousingDataset.csv")

# Step 3: Data Preprocessing
@task(log_prints=True)
def preprocess_data(df):
    df = df.drop('Order', axis=1)
    df = df.drop('PID', axis=1)
    
    for column in df.columns:
        if df[column].dtype in [np.int64, np.float64]:  # Numeric columns
            df[column].fillna(df[column].mean(), inplace=True)
        elif df[column].dtype == 'object':  # Categorical columns
            df[column].fillna(df[column].mode()[0], inplace=True)
            
            dummies_column = pd.get_dummies(df[column], prefix=column, dtype=int)
            df = pd.concat([df, dummies_column], axis=1)
            df = df.drop(column, axis=1)

    return df

# Step 4: Model Training
@task
def train_model(df):
    # Train your machine learning model with Logistic Regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import median_absolute_error
    
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = median_absolute_error(y_test, y_pred)
    
    return accuracy

# Step 5: Define Prefect Flow
# @flow(log_prints=True)
def workflow_ames_housing():
    # step 1 = loading data
    data = load_dataset()
    # step 2 = preprocessing
    preprocessed_data = preprocess_data(data)
    # step 3 = data modeling
    accuracy = train_model(preprocessed_data)

    print("Accuracy: ", accuracy)
   
# Step 6: Run the Prefect Flow
if __name__ == "__main__":
    workflow_ames_housing()
    # workflow_ames_housing.serve(name="ames-housing-workflow",
    #                   tags=["first workflow"],
    #                   parameters={},
    #                   interval=120)