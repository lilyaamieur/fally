import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# 1. CONFIG
# =========================
BASE_DIR = "./data"
FOLDERS = ["Healthy", "Light Damage", "Medium Damage", "Severe Damage"]
CLASS_LABELS = ["Healthy", "Light", "Medium", "Severe"]

# yeah yeah, Signal Processing is annoying, and we need to save both the model and scaler :/
MODEL_PATH = "holes_ensemble_classifier.pkl"
SCALER_PATH = "holes_scaler.pkl"

TRAIN_ENSEMBLE = True
ENSEMBLE_MODEL_TYPES = ['random_forest', 'svm', 'mlp']

# =========================
# 2. UTILS
# =========================
def read_signal_file(filepath):
    try:
        df = pd.read_csv(filepath, sep="\t", encoding="latin1", names=["Frequency", "Magnitude", "Phase"], header=0)
        magnitude = df["Magnitude"].values.astype(np.float32)

        magnitude = magnitude[np.isfinite(magnitude)]

        if len(magnitude) == 0:
            raise ValueError("No valid magnitude data found")

        return magnitude
    except Exception as e:
        print(f"Error reading {filepath}: {str(e)}")
        return None

# =========================
# 3. DATA LOADING
# =========================
def load_dataset(base_dir, folders):
    samples = []
    labels = []
    min_length = float('inf')

    print(f"Loading data from {base_dir}...")

    all_magnitudes = []
    all_labels = []

    for label, folder in enumerate(tqdm(folders, desc="Loading folders")):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' not found. Skipping.")
            continue

        for fname in os.listdir(folder_path):
            if fname.endswith(".txt"):
                path = os.path.join(folder_path, fname)
                mag = read_signal_file(path)
                if mag is not None:
                    all_magnitudes.append(mag)
                    all_labels.append(label)
                    min_length = min(min_length, len(mag))

    print(f"Truncating all signals to length: {min_length}")

    for mag, label in zip(all_magnitudes, all_labels):
        truncated_mag = mag[:min_length]
        samples.append(truncated_mag)
        labels.append(label)

    X = np.array(samples)
    y = np.array(labels)

    print(f"Loaded {len(X)} samples across {len(folders)} categories.")
    print(f"Signal length: {X.shape[1]}")

    if len(X) == 0:
        raise ValueError("No samples loaded. Check BASE_DIR and folder structure.")

    return X, y

# =========================
# 4. MODEL CREATION
# =========================
def create_individual_model(model_type: str) -> object:
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10
        )
    elif model_type == 'svm':
        return SVC(
            kernel='rbf',
            random_state=42,
            probability=True,
            C=1.0,
            gamma='scale'
        )
    elif model_type == 'mlp':
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate_init=0.001
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# =========================
# 5. TRAINING SETUP
# =========================
print("\nPreparing dataset...")
X, y = load_dataset(BASE_DIR, FOLDERS)

if len(X) == 0:
    print("No data found or loaded. Please check BASE_DIR and folder contents.")
    exit()

input_size = X.shape[1]
num_classes = len(FOLDERS)

print(f"Dataset loaded. Total samples: {len(X)}")
print(f"Input signal length: {input_size}")
print(f"Number of output classes: {num_classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if TRAIN_ENSEMBLE:
    print(f"Creating ensemble models: {', '.join(ENSEMBLE_MODEL_TYPES)}...")
    estimators = []
    trained_models = {}
    for model_type_name in ENSEMBLE_MODEL_TYPES:
        estimator = create_individual_model(model_type_name)
        estimators.append((model_type_name, estimator))
        trained_models[model_type_name] = estimator
    
    model = VotingClassifier(
        estimators=estimators, 
        voting='soft',
        n_jobs=-1
    )
else:
    print("Training a single RandomForestClassifier...")
    model = create_individual_model('random_forest')
    trained_models = {'random_forest': model}


if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print(f"Loading pre-existing ensemble model from {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        loaded_ensemble_or_model = pickle.load(f)
        if TRAIN_ENSEMBLE:
            model = loaded_ensemble_or_model
            for est_name, est_obj in model.estimators_:
                if est_name in trained_models:
                    trained_models[est_name] = est_obj
        else:
            model = loaded_ensemble_or_model
            trained_models = {'random_forest': model}
            
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully.")
else:
    print("No pre-existing model found. Training from scratch.")

# =========================
# 6. TRAINING
# =========================
print(f"\nStarting training...")
if TRAIN_ENSEMBLE:
    print("Training individual models for the ensemble...")
    for name, estimator in tqdm(model.estimators, desc="Fitting individual estimators"):
        print(f"  Fitting {name}...")
        estimator.fit(X_train_scaled, y_train)
    model.fit(X_train_scaled, y_train)
else:
    print(f"Training single model: {model.__class__.__name__}...")
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
# 9. INFERENCE
# =========================
def predict_observation(file_path, trained_model, trained_scaler, input_feature_size, class_labels):
    mag = read_signal_file(file_path)
    if mag is None:
        print(f"Failed to read or process '{file_path}'. Cannot make prediction.")
        return

    if len(mag) < input_feature_size:
        mag = np.pad(mag, (0, input_feature_size - len(mag)), 'constant')
    elif len(mag) > input_feature_size:
        mag = mag[:input_feature_size]

    mag_reshaped = mag.reshape(1, -1)
    mag_scaled = trained_scaler.transform(mag_reshaped)

    if hasattr(trained_model, 'predict_proba'):
        probabilities = trained_model.predict_proba(mag_scaled)[0]
    else:
        prediction_idx = trained_model.predict(mag_scaled)[0]
        probabilities = np.zeros(len(class_labels))
        probabilities[prediction_idx] = 1.0

    print(f"\nPrediction for '{file_path}':")
    for i, p in enumerate(probabilities):
        print(f" {class_labels[i]}: {p*100:.2f}%")

    predicted_label_idx = np.argmax(probabilities)
    print(f"Final Prediction: {class_labels[predicted_label_idx]}")

# ============ Example Usage ============
print("\n--- Running Inference on 'Observation.txt' ---")
if os.path.exists("Observation.txt"):
    predict_observation("Observation.txt", model, scaler, input_size, CLASS_LABELS)
else:
    print("Observation.txt not found. Creating a dummy file for testing...")
    with open("Observation.txt", "w") as f:
        f.write("Frequency\tMagnitude\tPhase\n")
        for i in range(input_size):
            f.write(f"{i}\t{np.random.rand()}\t{np.random.rand()}\n")
    print("Dummy file created. Running prediction...")
    predict_observation("Observation.txt", model, scaler, input_size, CLASS_LABELS)

print("\n--- Holes Model Training and Evaluation Complete ---")
