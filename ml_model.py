import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_data(filepath):
    """
    Loads and preprocesses the dataset from a CSV file.

    Parameters:
    - filepath (str): Path to the CSV file containing the dataset.

    Returns:
    - X (pd.Series): Preprocessed symptom data.
    - y (pd.Series): Corresponding disease labels.
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Fill NaN values with empty strings
    df = df.fillna("")
    
    # Initialize the 'Symptom' column
    df['Symptom'] = ""
    
    # Concatenate all symptom columns into the 'Symptom' column
    for i in range(1, 18):
        symptom_col = f"Symptom_{i}"
        df['Symptom'] += df[symptom_col] + " "
    
    # Remove trailing spaces
    df['Symptom'] = df['Symptom'].str.strip()
    
    # Drop the original individual symptom columns
    symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]
    df = df.drop(columns=symptom_columns)
    
    # Define features and target
    X = df['Symptom']
    y = df['Disease']
    
    return X, y



def train_model(X, y, test_size=0.25, random_state=44):
    """
    Splits the data, trains the model, and evaluates its performance.

    Parameters:
    - X (pd.Series): Preprocessed symptom data.
    - y (pd.Series): Corresponding disease labels.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    - text_clf (Pipeline): Trained machine learning pipeline.
    - report (str): Classification report on the test data.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    # Define the pipeline with TF-IDF Vectorizer and LinearSVC
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC()),
    ])
    
    # Train the model
    text_clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = text_clf.predict(X_test)
    
    # Generate a classification report
    report = classification_report(y_test, predictions)
    
    print("Model Training Complete.")
    print("Classification Report:")
    print(report)
    
    return text_clf, report


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


def save_model(model, filepath):
    """
    Saves the trained model to a file.

    Parameters:
    - model (Pipeline): Trained machine learning pipeline.
    - filepath (str): Path where the model will be saved.
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}.")

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


def main():
    # Path to your dataset
    dataset_path = 'dataset.csv'
    
    # Preprocess the data
    X, y = preprocess_data(dataset_path)
    
    # Train the model
    model, report = train_model(X, y)
    
    # Optional: Save the trained model for future use
    model_path = 'disease_prediction_model.joblib'
    save_model(model, model_path)
    
    # # Optional: Load the model (if needed)
    # # model = load_model(model_path)
    
    # # Example input symptoms
    # user_symptoms = [
    #     "headache",
    #     "fever",
    #     "nausea",
    #     "dizziness"
    # ]
    
    # # Predict the disease
    # predicted_disease = predict_disease(model, user_symptoms)
    # print(f"\nPredicted Disease: {predicted_disease}")

if __name__ == "__main__":
    main()
