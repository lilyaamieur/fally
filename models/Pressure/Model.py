import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# 1. CONFIG
# =========================
# IMPORTANT: BASE_DIR should point to the directory containing your stress data folders
# e.g., if your folders are in './my_stress_data/healthy', './my_stress_data/light_stress', etc.
BASE_DIR = "./data" # Make sure this matches your data generation script's output_directory

# UPDATED FOLDERS and CLASS_LABELS for the new stress conditions
FOLDERS = ["healthy", "light_stress", "medium_stress", "heavy_stress"]
CLASS_LABELS = ["Healthy", "Light Stress", "Medium Stress", "Heavy Stress"]

# Model configuration
MODEL_PATH = "stress_classifier.pkl"  # Changed from .pth to .pkl
SCALER_PATH = "stress_scaler.pkl"     # Path for the scaler

# This MUST match the FIXED_SIGNAL_LENGTH used in your data generation script
FIXED_SIGNAL_LENGTH = 6402

# Model selection - choose one of: 'random_forest', 'svm', 'mlp'
MODEL_TYPE = 'random_forest'

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
        print(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        return None

# =========================
# 3. DATA LOADING
# =========================
def load_dataset(base_dir: str, folders: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the dataset from the specified directories.
    Returns features (X) and labels (y) as numpy arrays.
    """
    samples = []
    labels = []
    print(f"Loading data from {base_dir}...")
    
    for label, folder in enumerate(tqdm(folders, desc="Loading folders")):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' not found. Skipping.")
            continue

        for fname in os.listdir(folder_path):
            if fname.endswith(".txt"):
                path = os.path.join(folder_path, fname)
                mag = read_signal_file(path)
                if mag is not None and len(mag) == FIXED_SIGNAL_LENGTH:
                    samples.append(mag)
                    labels.append(label)
                else:
                    print(f"Skipping file {fname} due to read error or incorrect length.")

    X = np.array(samples)
    y = np.array(labels)
    print(f"Loaded {len(X)} samples across {len(folders)} categories.")
    
    if len(X) == 0:
        raise ValueError("No samples loaded. Check BASE_DIR and folder structure.")
    
    return X, y

# =========================
# 4. MODEL CREATION
# =========================
def create_model(model_type: str, input_size: int) -> object:
    """
    Creates and returns a scikit-learn model based on the specified type.
    """
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'svm':
        return SVC(
            kernel='rbf',
            random_state=42,
            probability=True  # Enable probability estimates
        )
    elif model_type == 'mlp':
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# =========================
# 5. TRAINING SETUP
# =========================
print("\nPreparing dataset...")
X, y = load_dataset(BASE_DIR, FOLDERS)

# Check if any data was loaded
if len(X) == 0:
    print("No data found or loaded. Please check BASE_DIR and folder contents.")
    exit()

input_size = X.shape[1]  # Get the signal length
num_classes = len(FOLDERS)

print(f"Dataset loaded. Total samples: {len(X)}")
print(f"Input signal length: {input_size}")
print(f"Number of output classes: {num_classes}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features for better performance (especially important for SVM and MLP)
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize model
print(f"Creating {MODEL_TYPE} model...")
model = create_model(MODEL_TYPE, input_size)

# Load model and scaler if they exist
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print(f"Loading pre-existing model from {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully.")
else:
    print("No pre-existing model found. Training from scratch.")
    
    # =========================
    # 6. TRAINING
    # =========================
    print(f"\nStarting training with {MODEL_TYPE}...")
    model.fit(X_train_scaled, y_train)
    print("Training completed!")

# =========================
# 7. EVALUATION
# =========================
print("\nEvaluating model performance...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

# =========================
# 8. SAVE MODEL AND SCALER
# =========================
print(f"Saving model to {MODEL_PATH}")
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"Saving scaler to {SCALER_PATH}")
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")

# =========================
# 9. INFERENCE ON A SINGLE FILE
# =========================
def predict_observation(file_path: str):
    """
    Predicts the stress category for a single observation file.
    """
    # Load and preprocess the observation file
    mag = read_signal_file(file_path)
    if mag is None:
        print(f"Failed to read or process '{file_path}'. Cannot make prediction.")
        return

    if len(mag) != FIXED_SIGNAL_LENGTH:
        print(f"Warning: Input file '{file_path}' has length {len(mag)}, expected {FIXED_SIGNAL_LENGTH}. "
              "Prediction might be inaccurate if padding/truncation changed the signal content significantly.")
    
    # Reshape for single sample prediction and scale
    mag_reshaped = mag.reshape(1, -1)
    mag_scaled = scaler.transform(mag_reshaped)
    
    # Get prediction and probabilities
    prediction = model.predict(mag_scaled)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(mag_scaled)[0]
    else:
        # For models without probability support, create a simple confidence score
        probabilities = np.zeros(len(CLASS_LABELS))
        probabilities[prediction] = 1.0

    print(f"\nPrediction for '{file_path}':")
    for i, p in enumerate(probabilities):
        print(f"  {CLASS_LABELS[i]}: {p*100:.2f}%")
    
    print(f"Final Prediction: {CLASS_LABELS[prediction]}")

# ============ Example Usage ============
# Make sure "Observation.txt" exists in your current working directory
# or provide the full path to the file.

print("\n--- Running Inference on 'Observation.txt' ---")
if os.path.exists("Observation.txt"):
    predict_observation("Observation.txt")
else:
    print("Observation.txt not found. Skipping inference example.")

# Alternatively, if you want to test a specific file from your generated dataset:
# predict_observation(os.path.join(BASE_DIR, 'healthy', 'healthy_0001.txt'))
# predict_observation(os.path.join(BASE_DIR, 'heavy_stress', 'heavy_stress_0001.txt'))

print("\n--- Model Training and Evaluation Complete ---")