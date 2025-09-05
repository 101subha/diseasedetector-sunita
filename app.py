from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib

# Load model (update path if needed)
model = joblib.load("disease_model.joblib")

# FastAPI app
app = FastAPI()

# Input schema for JSON
class SymptomsInput(BaseModel):
    symptoms: list[str]

@app.get("/")
def home():
    return {"message": "Disease Detector API is running!"}

# ✅ 1. POST endpoint (for Lovable, Postman, etc.)
@app.post("/predict")
def predict(input_data: SymptomsInput):
    symptoms = input_data.symptoms
    prediction = model.predict([symptoms])  # dummy example
    return {"prediction": prediction[0]}

# ✅ 2. GET endpoint (for browser testing with query params)
@app.get("/predict")
def predict_query(symptoms: str = Query(..., description="Comma separated symptoms")):
    symptoms_list = symptoms.split(",")
    prediction = model.predict([symptoms_list])  # dummy example
    return {"prediction": prediction[0]}
