import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
            return None
            
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
    target_variable = "quality"
    
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset.")

    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    
    return X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains multiple models, performs hyperparameter tuning, and returns the best one.
    Also handles feature scaling via Pipeline.
    """
    
    # Define base models to test
    # We use Pipelines to ensure scaling is applied correctly (fit on train, transform on test)
    pipelines = {
        "RandomForest": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(random_state=RANDOM_STATE))
        ]),
        "GradientBoosting": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(random_state=RANDOM_STATE))
        ]),
        "LinearRegression": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        "ElasticNet": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ElasticNet(random_state=RANDOM_STATE))
        ])
    }

    # Define Hyperparameter Grids
    # Keeping it relatively small for Lab demo speed, but enough to show tuning
    param_grids = {
        "RandomForest": {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5]
        },
        "GradientBoosting": {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.05, 0.1],
            'regressor__max_depth': [3, 5]
        },
        "LinearRegression": {}, # No hyperparameters to tune usually
        "ElasticNet": {
            'regressor__alpha': [0.1, 1.0],
            'regressor__l1_ratio': [0.5, 0.7]
        }
    }

    best_model = None
    best_metrics = {"R2_Score": -float("inf")}
    best_name = ""

    print("\nTraining, Tuning, and Evaluating models...")
    print("-" * 95)
    print(f"{'Model':<20} | {'MSE':<10} | {'RMSE':<10} | {'MAE':<10} | {'R2 Score':<10} | {'Tuned?':<8}")
    print("-" * 95)

    for name, pipeline in pipelines.items():
        # Perform Grid Search if params exist
        grid = param_grids.get(name)
        tuned = False
        
        if grid:
            search = GridSearchCV(pipeline, grid, cv=3, scoring='r2', n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            tuned = True
            # print(f"  Best params for {name}: {search.best_params_}")
        else:
            model = pipeline
            model.fit(X_train, y_train)

        # Evaluate on Test Set
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        tuned_str = "Yes" if tuned else "No"
        print(f"{name:<20} | {mse:<10.4f} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f} | {tuned_str:<8}")
        
        if r2 > best_metrics["R2_Score"]:
            best_metrics = {
                "MSE": mse, 
                "RMSE": rmse,
                "MAE": mae,
                "R2_Score": r2
            }
            best_model = model
            best_name = name

    print("-" * 95)
    print(f"Best Model: {best_name}")
    return best_model, best_metrics

def save_artifacts(model, metrics, model_path, metrics_path):
    """
    Saves the trained model and evaluation metrics to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

def main():
    df = load_data(DATA_PATH)
    if df is None:
        exit(1)

    X, y = preprocess_and_select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    best_model, metrics = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("\nBest Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    save_artifacts(best_model, metrics, MODEL_SAVE_PATH, METRICS_SAVE_PATH)

if __name__ == "__main__":
    main()
