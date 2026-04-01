import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging
import warnings

# Suppress benign MLflow and Scikit-Learn warnings in terminal
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# --- Configuration ---
STUDENT_NAME = "Madhav Murali"
ROLL_NUMBER = "2022BCS0050"
EXPERIMENT_NAME = "2022bcs0050_experiment"

# Explicitly use an unambiguous Local SQLite backend to securely sync tracking states
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "../mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# Set MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_V1_PATH = os.path.join(script_dir, "../data/dataset_v1.csv")
DATA_V2_PATH = os.path.join(script_dir, "../data/dataset_v2.csv")
METRICS_OUTPUT = os.path.join(script_dir, "../outputs/metrics.json")

# Make outputs directory if it doesn't exist
os.makedirs(os.path.join(script_dir, "../outputs"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "../models"), exist_ok=True)

def load_data(path, feature_selection=False):
    """ Loads data from a CSV, optionally applying feature selection. """
    # Data is separated by semicolon
    df = pd.read_csv(path, sep=";")
    
    target_variable = "quality"
    y = df[target_variable]
    X = df.drop(target_variable, axis=1)

    if feature_selection:
        # Mandatory feature selection requirement:
        # We drop some supposedly less important features for simplicity
        features_to_drop = ["chlorides", "free sulfur dioxide", "pH"]
        X = X.drop(columns=[c for c in features_to_drop if c in X.columns])
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log(run_name, data_path, model, is_v2, is_feature_selection=False):
    """ Train model, log params and metrics to MLflow, and return metrics. """
    print(f"\n--- Starting {run_name} ---")
    
    # Enable active MLflow run
    with mlflow.start_run(run_name=run_name):
        X_train, X_test, y_train, y_test = load_data(data_path, is_feature_selection)

        # Log parameters
        mlflow.log_param("dataset_version", "v2" if is_v2 else "v1")
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("feature_selection_applied", is_feature_selection)
        
        # If Random Forest, log n_estimators
        if isinstance(model, RandomForestRegressor):
            mlflow.log_param("n_estimators", model.n_estimators)

        # Train model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged MSE: {mse:.4f}, R2: {r2:.4f}")
        return {
            "run_name": run_name,
            "mse": mse,
            "r2_score": r2,
            "dataset": "v2" if is_v2 else "v1",
            "model_type": type(model).__name__
        }

def main():
    results = []
    
    # Ensure datasets exist, falling back to full dataset if not created by DVC yet
    if not os.path.exists(DATA_V1_PATH) or not os.path.exists(DATA_V2_PATH):
        print(f"Creating missing dataset files directly from winequality-red.csv...")
        full_df = pd.read_csv(os.path.join(script_dir, "../data/winequality-red.csv"), sep=";")
        full_df.head(501).to_csv(DATA_V1_PATH, sep=";", index=False)
        full_df.to_csv(DATA_V2_PATH, sep=";", index=False)

    # Run 1: Version 1, Model A (Random Forest Baseline)
    r1 = train_and_log(
        run_name="Run 1",
        data_path=DATA_V1_PATH,
        model=RandomForestRegressor(n_estimators=100, random_state=42),
        is_v2=False
    )
    results.append(r1)

    # Run 2: Version 1, Model A (Hyperparameter changed)
    r2 = train_and_log(
        run_name="Run 2",
        data_path=DATA_V1_PATH,
        model=RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        is_v2=False
    )
    results.append(r2)

    # Run 3: Version 2, Model A (Baseline Random Forest)
    # Save the best Model A for Docker API to pick up later (models/model.pkl)
    model3 = RandomForestRegressor(n_estimators=100, random_state=42)
    r3 = train_and_log(
        run_name="Run 3",
        data_path=DATA_V2_PATH,
        model=model3,
        is_v2=True
    )
    import joblib
    joblib.dump(model3, os.path.join(script_dir, "../models/model.pkl")) 
    results.append(r3)

    # Run 4: Version 2, Model A (Feature selection)
    r4 = train_and_log(
        run_name="Run 4",
        data_path=DATA_V2_PATH,
        model=RandomForestRegressor(n_estimators=100, random_state=42),
        is_v2=True,
        is_feature_selection=True
    )
    results.append(r4)

    # Run 5: Version 2, Model B (Linear Regression) + Feature selection
    r5 = train_and_log(
        run_name="Run 5",
        data_path=DATA_V2_PATH,
        model=LinearRegression(),
        is_v2=True,
        is_feature_selection=True
    )
    results.append(r5)

    # Formatting required output JSON file
    output_metrics = {
        "Name": STUDENT_NAME,
        "Roll No": ROLL_NUMBER,
        "Experiments": results
    }

    with open(METRICS_OUTPUT, "w") as f:
        json.dump(output_metrics, f, indent=4)
        print(f"\nSaved metrics to {METRICS_OUTPUT}")

if __name__ == "__main__":
    main()
