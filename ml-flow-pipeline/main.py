# Installations
# 1. pip install mlflow
# 2. pip install psutil

# Steps to run
# 1. In terminal, run command -> mlflow ui --host 0.0.0.0 --port 5001
# 2. Right click on main.py and "run in interactive terminal"
# 3. Open localhost:5001 in browser and see the experimental results

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error
import mlflow
import mlflow.sklearn
import psutil
import time

# Set the MLflow tracking URI to 'http'
mlflow.set_tracking_uri("http://localhost:5001")

# Function for data preprocessing
def preprocess_data(data):
    # Drop Order and PID columns as they are just indexes
    # Drop other columns which have a lot of empty values
    data = data.drop(columns=['Order', 'PID', 'Pool QC', 'Alley', 'Misc Feature'])
    
    scaler = MinMaxScaler()
    
    for column in data.columns:
        if column == 'SalePrice':
            continue
        
        if data[column].dtype in [np.int64, np.float64]:  
            # Numeric columns fill empty with mean
            data[column].fillna(data[column].mean(), inplace=True)
            
            # normalize data with MinMaxScaler
            data[column] = scaler.fit_transform(data[[column]])
        elif data[column].dtype == 'object':  
            # Fill NaN values with Mode val
            data[column].fillna(data[column].mode(), inplace=True)
            
            # Convert categorical variables to one-hot encoding
            dummies_column = pd.get_dummies(data[column], prefix=column, dtype=int)
            data = pd.concat([data, dummies_column], axis=1)
            data = data.drop(column, axis=1)
    
    # Split data into X (features) and y (target)
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']

    return X, y

# Function for training the model
def train_model(X_train, y_train, max_depth=4, n_estimators=100):
    # Initialize the regressor model
    clf = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    return clf

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate MSQ
    mean_abs_error = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mean_abs_error}")

    # Display Root Mean Squared Error report
    print("Root Mean Sqaured Error:")
    print(root_mean_squared_error(y_test, y_pred))
    
    # Display R2 Score report
    print("R2 Score:")
    print(r2_score(y_test, y_pred))
    
    # Display Median Absolute Error report
    print("Median Absolute Error Score:")
    print(median_absolute_error(y_test, y_pred))

# Function to log model and system metrics to MLflow
def log_to_mlflow(model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Log hyper parameters using in Random Forest Algorithm
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("n_estimators", model.n_estimators)

        # Log model metrics
        y_pred = model.predict(X_test)
        mAbsError = mean_absolute_error(y_test, y_pred)
        rMSqError = root_mean_squared_error(y_test, y_pred)
        medianAbsError = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metric("mean-absolute-error", mAbsError)
        mlflow.log_metric("root-mean-squared-error", rMSqError)
        mlflow.log_metric("median-absolute-error", medianAbsError)
        mlflow.log_metric("r2-score", r2)
        
        # Log system metrics
        # Example: CPU and Memory Usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        mlflow.log_metric("system_cpu_usage", cpu_usage)
        mlflow.log_metric("system_memory_usage", memory_usage)

        # Log execution time for training the model
        execution_time = {}  # Dictionary to store execution times for different stages
        # Example: Execution time for training the model
        start_time = time.time()
        model = train_model(X_train, y_train)
        end_time = time.time()
        execution_time["system_model_training"] = end_time - start_time

        # Log execution time 
        mlflow.log_metrics(execution_time)

        # Evaluate model and log metrics
        evaluate_model(model, X_test, y_test)

        # Log model
        mlflow.sklearn.log_model(model, "model")

def load_dataset():
    return pd.read_csv("../dataset/AmesHousingDataset.csv")

# Main function
def main():
    # Load the dataset
    data = load_dataset()

    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate and log to MLflow
    log_to_mlflow(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()