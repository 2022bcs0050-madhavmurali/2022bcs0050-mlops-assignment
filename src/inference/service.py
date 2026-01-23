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
    name: str
    roll_no: str
    wine_quality: float


@app.get("/")
def root():
    return {"message": "Wine Quality Prediction API", "student": "Madhav Murali", "roll_no": "2022BCS0050"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: WineQualityInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return PredictionResponse(
        name="Madhav Murali",
        roll_no="2022BCS0050",
        wine_quality=round(float(prediction), 2)
    )
