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
    """
    if not isinstance(input_symptoms, list):
        raise ValueError("Input symptoms should be provided as a list of strings.")
    
    # Concatenate the list of symptoms into a single string separated by spaces
    symptoms_text = ' '.join(input_symptoms)
    
    # Use the model to predict the disease
    prediction = model.predict([symptoms_text])[0]
    
    return prediction

def explain_disease_generativeai(disease_name):
    """
    Uses Google Generative AI's API to provide a detailed explanation of the predicted disease.
    
    Parameters:
    - disease_name (str): The name of the disease to explain.
    
    Returns:
    - explanation (str): Detailed explanation of the disease.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"Provide a detailed explanation of the disease '{disease_name}', including:\n"
        f"■ Common symptoms\n"
        f"■ Possible causes\n"
        f"■ Recommended treatments and next steps\n"
    )
    
    try:
        response = model.generate_content(prompt)
        
        explanation = response.text.strip()
        return explanation if explanation else "No explanation received."
    
    except Exception as e:
        return f"An error occurred while fetching the explanation: {str(e)}"

def main():
    # Configure Google Generative AI
    configure_generativeai()
    
    model_path = "disease_prediction_model.joblib"
    model = load_model(model_path)
    
    while True:
        print("\nEnter the symptoms separated by commas (e.g., headache, fever, nausea) or type 'exit' to quit:")
        user_input = input("Symptoms: ")
        
        if user_input.lower() == 'exit':
            print("Exiting the prediction system.")
            break
        
        # Split the input string into a list of symptoms
        user_symptoms = [symptom.strip().lower() for symptom in user_input.split(',') if symptom.strip()]
        
        if not user_symptoms:
            print("No valid symptoms entered. Please try again.")
            continue
        
        # Predict the disease
        try:
            predicted_disease = predict_disease(model, user_symptoms)
            print(f"\nPredicted Disease: {predicted_disease}")
        except Exception as e:
            print(f"An error occurred during prediction: {str(e)}")
            continue
        
        # Get detailed explanation from Google Generative AI
        explanation = explain_disease_generativeai(predicted_disease)
        print("\nDisease Explanation:")
        print(explanation)

if __name__ == "__main__":
    main()
