import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import sys
import json

# =========================
# 1. CONFIG
# =========================
# Paths to the saved model and scaler files
MODEL_PATH = "holes_ensemble_classifier.pkl"
SCALER_PATH = "holes_scaler.pkl"

# Labels corresponding to the classes the model was trained on
CLASS_LABELS = ["Healthy", "Light", "Medium", "Severe"]

# =========================
# 2. UTILS
# =========================
def read_signal_file(filepath: str) -> np.ndarray | None:
    """
    Reads a signal file (tab-separated) and extracts the Magnitude data.

    Args:
        filepath (str): The path to the .txt signal file.

    Returns:
        np.ndarray | None: A numpy array of magnitude values if successful,
                           otherwise None if there's an error or no valid data.
    """
    try:
        # Read the CSV file using pandas, specifying tab separator and column names
        df = pd.read_csv(filepath, sep="\t", encoding="latin1", names=["Frequency", "Magnitude", "Phase"], header=0)
        # Extract magnitude values and convert to float32 numpy array
        magnitude = df["Magnitude"].values.astype(np.float32)

        # Filter out any non-finite values (NaN, Inf) from the magnitude array
        magnitude = magnitude[np.isfinite(magnitude)]

        # If no valid magnitude data remains after filtering, raise an error
        if len(magnitude) == 0:
            raise ValueError("No valid magnitude data found in the file.")

        return magnitude
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"Error reading and processing '{filepath}': {str(e)}")
        return None

def predict_damage_class(file_path: str) -> dict | None:
    """
    Loads the trained model and scaler, then predicts the damage class
    and confidence for a given signal file.

    Args:
        file_path (str): The path to the .txt signal file to predict on.

    Returns:
        dict | None: A dictionary containing the predicted 'class' (str)
                     and 'confidence' (float), or None if prediction fails.
                     Confidence is the probability of the predicted class.
    """
    # Check if model and scaler files exist before proceeding
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please ensure the model is trained and saved.")
        return None
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler file not found at '{SCALER_PATH}'. Please ensure the scaler is trained and saved.")
        return None

    try:
        # Load the trained model using pickle
        with open(MODEL_PATH, 'rb') as f:
            trained_model = pickle.load(f)

        # Load the trained scaler using pickle
        with open(SCALER_PATH, 'rb') as f:
            trained_scaler = pickle.load(f)

    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        return None

    # Determine the expected input feature size from the loaded scaler
    # The n_features_in_ attribute holds the number of features seen during fit.
    try:
        input_feature_size = trained_scaler.n_features_in_
    except AttributeError:
        print("Error: Could not determine input feature size from scaler. Is the scaler fitted?")
        return None

    # Read the magnitude data from the input file
    mag = read_signal_file(file_path)
    if mag is None:
        return None # read_signal_file already printed the error

    # Pre-process the magnitude data to match the expected input_feature_size
    # If the signal is shorter, pad it with zeros.
    if len(mag) < input_feature_size:
        mag = np.pad(mag, (0, input_feature_size - len(mag)), 'constant')
    # If the signal is longer, truncate it.
    elif len(mag) > input_feature_size:
        mag = mag[:input_feature_size]

    # Reshape the magnitude array for single observation prediction (1 sample, n_features)
    mag_reshaped = mag.reshape(1, -1)

    # Scale the reshaped magnitude data using the loaded scaler
    mag_scaled = trained_scaler.transform(mag_reshaped)

    # Make a prediction
    # If the model has predict_proba (like ensemble, SVM with probability=True, MLP), use it for probabilities
    if hasattr(trained_model, 'predict_proba'):
        probabilities = trained_model.predict_proba(mag_scaled)[0]
    # Otherwise, fall back to predict and assign 100% to the predicted class
    else:
        prediction_idx = trained_model.predict(mag_scaled)[0]
        probabilities = np.zeros(len(CLASS_LABELS))
        probabilities[prediction_idx] = 1.0

    # Get the index of the class with the highest probability
    predicted_label_idx = np.argmax(probabilities)
    # Get the predicted class label string
    predicted_class = CLASS_LABELS[predicted_label_idx]
    # Get the confidence (probability) for the predicted class
    confidence = probabilities[predicted_label_idx]

    # Return the result as a dictionary
    return {
        "class": predicted_class,
        "confidence": float(confidence) # Ensure confidence is a standard float
    }

# =========================
# 3. MAIN EXECUTION
# =========================
def run_prediction(file_path):
    result = predict_damage_class(file_path)
    if result:
        return {
            "class": result["class"],
            "confidence": round(result["confidence"], 4)
        }
    return {
        "error": "Prediction failed"
    }

# print(run_prediction(sys.argv[1]) if len(sys.argv) > 1 else {"error": "No file path provided"})