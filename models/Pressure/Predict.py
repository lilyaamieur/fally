import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import sys

# =========================
# 1. CONFIG (MUST MATCH TRAINING SCRIPT)
# =========================
# IMPORTANT: These paths and configurations MUST match what was used during training
MODEL_PATH = "stress_classifier.pkl"
SCALER_PATH = "stress_scaler.pkl"

# UPDATED FOLDERS and CLASS_LABELS for the new stress conditions
CLASS_LABELS = ["Healthy", "Light Stress", "Medium Stress", "Heavy Stress"]

# This MUST match the FIXED_SIGNAL_LENGTH used in your data generation script
FIXED_SIGNAL_LENGTH = 6402

# =========================
# 2. UTILS
# =========================
def read_signal_file(filepath: str) -> np.ndarray | None:
    """
    Reads a signal file and returns its Magnitude data,
    ensuring it's padded or truncated to FIXED_SIGNAL_LENGTH.
    Handles potential headers and bad lines.
    """
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            encoding="latin1",
            header=None, # Treat everything as data initially
            names=["Frequency", "Magnitude", "Phase"],
            on_bad_lines="skip"
        )

        # Convert to numeric, coercing errors to NaN and drop invalid rows
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if df.empty:
            raise ValueError("No valid numeric data found after parsing")
            
        magnitude = df["Magnitude"].values.astype(np.float32)

        # Pad or truncate the signal to the FIXED_SIGNAL_LENGTH
        if len(magnitude) < FIXED_SIGNAL_LENGTH:
            padded_magnitude = np.pad(magnitude, (0, FIXED_SIGNAL_LENGTH - len(magnitude)), 'constant')
            return padded_magnitude
        elif len(magnitude) > FIXED_SIGNAL_LENGTH:
            return magnitude[:FIXED_SIGNAL_LENGTH]
        else:
            return magnitude

    except Exception as e:
        # Error messages might be useful for debugging, but are removed as per request.
        # print(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        return None

# =========================
# 3. MODEL AND SCALER LOADING
# =========================
_model = None
_scaler = None

def load_artifacts():
    """
    Loads the pre-trained model and scaler.
    This function is called only once when the script is imported.
    """
    global _model, _scaler
    if _model is None or _scaler is None:
        try:
            # print(f"Loading model from {MODEL_PATH}...")
            with open(MODEL_PATH, 'rb') as f:
                _model = pickle.load(f)
            # print(f"Loading scaler from {SCALER_PATH}...")
            with open(SCALER_PATH, 'rb') as f:
                _scaler = pickle.load(f)
            # print("Model and scaler loaded successfully for prediction.")
        except FileNotFoundError:
            # print(f"Error: Model file '{MODEL_PATH}' or scaler file '{SCALER_PATH}' not found.")
            # print("Please ensure you have run the training script ('train_stress_model.py') first.")
            exit()
        except Exception as e:
            # print(f"An error occurred while loading artifacts: {e}")
            exit()

# Load artifacts when the script is imported
load_artifacts()

# =========================
# 4. PREDICTION FUNCTION
# =========================
def predict_stress_class(file_path: str) -> dict | None:
    """
    Predicts the stress category for a single observation file.
    Returns a dictionary with 'class' and 'confidence', or None if prediction fails.
    """
    # Load and preprocess the observation file
    mag = read_signal_file(file_path)
    if mag is None:
        # print(f"Failed to read or process '{file_path}'. Cannot make prediction.")
        return None

    if len(mag) != FIXED_SIGNAL_LENGTH:
        # print(f"Warning: Input file '{file_path}' has length {len(mag)}, expected {FIXED_SIGNAL_LENGTH}. "
        #       "Prediction might be inaccurate if padding/truncation changed the signal content significantly.")
        pass # Keep logic but remove print
    
    # Reshape for single sample prediction and scale
    mag_reshaped = mag.reshape(1, -1)
    mag_scaled = _scaler.transform(mag_reshaped)
    
    # Get prediction
    prediction_index = _model.predict(mag_scaled)[0]
    predicted_class = CLASS_LABELS[prediction_index]
    
    # Get probabilities if available
    if hasattr(_model, 'predict_proba'):
        probabilities = _model.predict_proba(mag_scaled)[0]
        confidence = probabilities[prediction_index]
    else:
        # For models without probability support, confidence is 1.0 for the predicted class
        confidence = 1.0 # Or you might choose to return a default low value like 0.5
        # print("Warning: Model does not support 'predict_proba'. Confidence will be 1.0 for the predicted class.")

    return {
        "class": predicted_class,
        "confidence": float(confidence) # Ensure confidence is a standard float
    }

# =========================
# 5. WRAPPER FOR EXTERNAL CALLS
# =========================
def run_prediction(file_path: str) -> dict:
    """
    Wrapper function to run the prediction and format the output.
    """
    result = predict_stress_class(file_path)
    if result:
        return {
            "class": result["class"],
            "confidence": round(result["confidence"], 4)
        }
    return {
        "error": "Prediction failed"
    }

# ============ Example Usage (for testing this script directly) ============
if __name__ == "__main__":
    # To run this example, make sure you have an 'Observation.txt' file
    # or specify a path to one of your generated signal files.
    example_file_path = "Observation.txt" 
    # example_file_path = os.path.join("./data", 'healthy', 'healthy_0001.txt') # Uncomment to test with a known file

    # print(f"\n--- Running example prediction for '{example_file_path}' ---")
    if os.path.exists(example_file_path):
        prediction_output = run_prediction(example_file_path)
        # print("Prediction Result:")
        # print(prediction_output)
    else:
        # print(f"Error: Example file '{example_file_path}' not found.")
        # print("Please create 'Observation.txt' or change 'example_file_path' to a valid signal file path.")
        pass # Keep logic but remove print

    # print("\n--- Prediction script finished ---")

print(run_prediction(sys.argv[1]) if len(sys.argv) > 1 else {"error": "No file path provided"})
