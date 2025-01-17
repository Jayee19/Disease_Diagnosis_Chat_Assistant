# api.py

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
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# After initializing the FastAPI app
app = FastAPI(title="Disease Prediction API", version="1.0")

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:5500",  # Example for VSCode Live Server
    "http://localhost:3000",    # Example for React dev server
    # Add other origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define Pydantic models for request and response

class PredictRequest(BaseModel):
    symptoms: List[str]

class PredictResponse(BaseModel):
    disease: str
    confidence: float

class ExplainRequest(BaseModel):
    disease: str

class ExplainResponse(BaseModel):
    explanation: str

def configure_generativeai():
    """
    Configures the Google Generative AI API key from environment variables.
    """
    api_key = os.getenv("GOOGLE_GENERATIVEAI_API_KEY")
    if not api_key:
        raise ValueError("Google Generative AI API key not found. Please set the GOOGLE_GENERATIVEAI_API_KEY environment variable.")
    genai.configure(api_key=api_key)

def load_model(filepath):
    """
    Loads a trained model from a file.

    Parameters:
    - filepath (str): Path to the saved model file.

    Returns:
    - model (Pipeline): Loaded machine learning pipeline.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at path: {filepath}")
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}.")
    return model

def predict_disease(model, input_symptoms):
    """
    Predicts the disease based on input symptoms using the trained model.

    Parameters:
    - model (Pipeline): Trained machine learning pipeline.
    - input_symptoms (list of str): List of symptoms provided as input.

    Returns:
    - prediction (str): Predicted disease.
    - confidence (float): Confidence score of the prediction.
    """
    if not isinstance(input_symptoms, list):
        raise ValueError("Input symptoms should be provided as a list of strings.")
    
    # Concatenate the list of symptoms into a single string separated by spaces
    symptoms_text = ' '.join(input_symptoms)
    
    # Use the model to predict the disease
    prediction = model.predict([symptoms_text])[0]
    
    # Get the confidence score using decision_function
    confidence_scores = model.decision_function([symptoms_text])
    
    # For multi-class classification, confidence_scores is an array of scores for each class
    # We take the score corresponding to the predicted class
    if isinstance(confidence_scores, np.ndarray):
        # Get the index of the predicted class
        class_index = np.where(model.classes_ == prediction)[0][0]
        confidence = float(confidence_scores[0][class_index])
    else:
        confidence = float(confidence_scores)
    
    return prediction, confidence

def explain_disease_generativeai(disease_name):
    """
    Uses Google Generative AI's API to provide a detailed explanation of the predicted disease.
    
    Parameters:
    - disease_name (str): The name of the disease to explain.
    
    Returns:
    - explanation (str): Detailed explanation of the disease.
    """
    prompt = (
        f"Provide a detailed explanation of the disease '{disease_name}', including:\n"
        f"■ Common symptoms\n"
        f"■ Possible causes\n"
        f"■ Recommended treatments and next steps\n"
    )
    
    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        explanation = response.text.strip()
        return explanation if explanation else "No explanation received."
    
    except Exception as e:
        return f"An error occurred while fetching the explanation: {str(e)}"

# Initialize the model at startup
@app.on_event("startup")
def startup_event():
    """
    Event handler that runs at application startup.
    Configures Generative AI and loads the machine learning model.
    """
    try:
        configure_generativeai()
    except ValueError as ve:
        print(ve)
        # Depending on requirements, you might want to stop the application here
    except Exception as e:
        print(f"Unexpected error during Generative AI configuration: {e}")
    
    global model
    try:
        model = load_model("disease_prediction_model.joblib")
    except FileNotFoundError as fnfe:
        print(fnfe)
        # Depending on requirements, you might want to stop the application here
    except Exception as e:
        print(f"Unexpected error during model loading: {e}")

# /predict endpoint
@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """
    Predicts the disease based on input symptoms.

    Parameters:
    - request (PredictRequest): Contains a list of symptoms.

    Returns:
    - PredictResponse: Predicted disease and confidence score.
    """
    try:
        prediction, confidence = predict_disease(model, request.symptoms)
        return PredictResponse(disease=prediction, confidence=confidence)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# /explain endpoint
@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(request: ExplainRequest):
    """
    Provides a detailed explanation of a disease.

    Parameters:
    - request (ExplainRequest): Contains the disease name.

    Returns:
    - ExplainResponse: Detailed explanation of the disease.
    """
    try:
        explanation = explain_disease_generativeai(request.disease)
        return ExplainResponse(explanation=explanation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
