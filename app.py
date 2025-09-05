from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Input schema
class SymptomInput(BaseModel):
    symptoms: list[str]

# Initialize FastAPI app
app = FastAPI(title="Disease Prediction API")

# Load model and encoders
model = joblib.load("disease_model.joblib")
mlb = joblib.load("symptom_encoder.joblib")
le = joblib.load("label_encoder.joblib")

@app.get("/")
def home():
    return {"message": "✅ Disease Prediction API is running!"}

@app.post("/predict")
def predict(data: SymptomInput):
    symptoms = data.symptoms

    # Filter unknown symptoms
    valid_symptoms = [s for s in symptoms if s in mlb.classes_]
    if not valid_symptoms:
        return {"error": "No valid symptoms provided"}

    # Encode input
    Xq = mlb.transform([valid_symptoms])

    # Make prediction
    pred = model.predict(Xq)[0]
    disease = le.inverse_transform([pred])[0]

    return {
        "input_symptoms": symptoms,
        "used_symptoms": valid_symptoms,
        "predicted_disease": disease
    }
