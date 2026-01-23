import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
DATA_PATH = "data/winequality-red.csv"
MODEL_SAVE_PATH = "models/model.pkl"
METRICS_SAVE_PATH = "metrics.json"
RANDOM_STATE = 42

def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    """
    try:
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            # Try to print current directory content to debug in CI
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory listing: {os.listdir('.')}")
            if os.path.exists('data'):
                print(f"Data directory listing: {os.listdir('data')}")
            return None
            
        # The notebook specified sep=";" for this dataset
        df = pd.read_csv(filepath, sep=";")
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_and_select_features(df):
    """
    Prepares features (X) and target (y).
    """
    # Define Target and Features
    target_variable = "quality"
    
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset.")

    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    
    return X, y

def train_model(X_train, y_train):
    """
    Trains the Random Forest Regressor.
    """
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using MSE and R2 Score.
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {"MSE": mse, "R2_Score": r2}

def save_artifacts(model, metrics, model_path, metrics_path):
    """
    Saves the trained model and evaluation metrics to disk.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save Model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save Metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

def main():
    # 1. Load the dataset
    df = load_data(DATA_PATH)
    if df is None:
        exit(1)

    # 2. Apply pre-processing and feature selection
    X, y = preprocess_and_select_features(df)

    # Split data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 3. Train the selected model
    print("Training Random Forest model...")
    model = train_model(X_train, y_train)

    # 4. Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    # 6. Print metrics to standard output
    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.6f}")
    print(f"R² Score: {metrics['R2_Score']:.6f}")

    # 5. Save model and metrics
    save_artifacts(model, metrics, MODEL_SAVE_PATH, METRICS_SAVE_PATH)

if __name__ == "__main__":
    main()
