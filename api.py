import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Disease Diagnosis Chat Assistant",
    description="An AI-powered disease diagnosis system that predicts diseases from symptoms",
    version="1.0"
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models
class PredictRequest(BaseModel):
    symptoms: str

class ExplainRequest(BaseModel):
    disease_name: str

# Helper functions
def configure_generativeai():
    api_key = os.getenv("GOOGLE_GENERATIVEAI_API_KEY")
    if not api_key:
        raise ValueError("Google Generative AI API key not found. Please set the GOOGLE_GENERATIVEAI_API_KEY environment variable.")
    genai.configure(api_key=api_key)

def load_model(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at path: {filepath}")
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}.")
    return model

def predict_disease(model, input_symptoms: str):
    if not isinstance(input_symptoms, str):
        raise ValueError("Input symptoms should be provided as a string.")
    
    symptoms_text = input_symptoms.lower()  # Convert to lowercase
    prediction = model.predict([symptoms_text])[0]
    confidence_scores = model.decision_function([symptoms_text])
    
    if isinstance(confidence_scores, np.ndarray):
        class_index = np.where(model.classes_ == prediction)[0][0]
        confidence = float(confidence_scores[0][class_index])
    else:
        confidence = float(confidence_scores)
    
    return prediction, confidence

def explain_disease_generativeai(disease_name):
    prompt = (
        f"Provide a detailed explanation of the disease '{disease_name}', including:\n"
        f"■ Common symptoms\n"
        f"■ Possible causes\n"
        f"■ Recommended treatments and next steps\n"
        f"Please note that this is for informational purposes only and not a substitute for professional medical advice."
    )
    
    try:
        model_genai = genai.GenerativeModel(model_name="gemini-pro")  # Fixed the model initialization
        response = model_genai.generate_content(prompt)
        explanation = response.text.strip()
        return explanation if explanation else "No explanation received."
    except Exception as e:
        # Return a more user-friendly error message
        return """Here is some general information about the condition:
        
Common symptoms may vary depending on the specific condition.
Please consult a healthcare provider for accurate diagnosis and treatment.
This is only a preliminary assessment based on the symptoms provided."""

# Global model variable
model = None

# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        configure_generativeai()
        global model
        model = load_model("disease_prediction_model.joblib")
    except Exception as e:
        print(f"Error during startup: {e}")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Disease Diagnosis Chat Assistant API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict disease from symptoms",
            "/explain": "POST - Get detailed disease explanation",
            "/docs": "GET - API documentation"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Predict endpoint
@app.post("/predict")
async def predict_endpoint(request: PredictRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prediction, confidence = predict_disease(model, request.symptoms)
        return {
            "predicted_disease": prediction,
            "confidence": confidence
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Explain endpoint
@app.post("/explain")
async def explain_endpoint(request: ExplainRequest):
    try:
        explanation = explain_disease_generativeai(request.disease_name)
        return {
            "disease_name": request.disease_name,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"error": exc.detail, "status_code": exc.status_code}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)