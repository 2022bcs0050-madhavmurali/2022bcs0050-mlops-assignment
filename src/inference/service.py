from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Wine Quality Prediction API")

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/model.pkl")
model = joblib.load(MODEL_PATH)


class WineQualityInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


class PredictionResponse(BaseModel):
    Name: str
    Roll_Number: str
    Prediction: float


@app.get("/")
def root():
    return {"status": "healthy", "Name": "Madhav Murali", "Roll No": "2022BCS0050"}

@app.get("/health")
def health():
    return {"status": "healthy", "Name": "Madhav Murali", "Roll No": "2022BCS0050"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: WineQualityInput):
    # Convert input to DataFrame with column name mapping
    # The model expects column names with spaces (as in the original dataset)
    input_dict = input_data.dict()
    
    # Map underscore names to space names
    column_mapping = {
        'fixed_acidity': 'fixed acidity',
        'volatile_acidity': 'volatile acidity',
        'citric_acid': 'citric acid',
        'residual_sugar': 'residual sugar',
        'free_sulfur_dioxide': 'free sulfur dioxide',
        'total_sulfur_dioxide': 'total sulfur dioxide'
    }
    
    # Create DataFrame with correct column names
    mapped_dict = {column_mapping.get(k, k): v for k, v in input_dict.items()}
    input_df = pd.DataFrame([mapped_dict])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return {
        "Name": "Madhav Murali",
        "Roll_Number": "2022BCS0050",
        "Prediction": round(float(prediction), 2)
    }
