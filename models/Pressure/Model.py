import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# 1. CONFIG
# =========================
BASE_DIR = "./data"
FOLDERS = ["healthy", "light_stress", "medium_stress", "heavy_stress"]
CLASS_LABELS = ["Healthy", "Light", "Medium", "Severe"]

MODEL_PATH = "holes_classifier_model.pkl"
SCALER_PATH = "holes_scaler.pkl"

TRAIN_BAGGING = False 

TRAIN_ENSEMBLE = True 

BAGGING_BASE_ESTIMATOR = 'decision_tree' 
BAGGING_N_ESTIMATORS = 50 

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
            raise ValueError("No valid magnitude data found after filtering non-finite values.")

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
    print(f"Signal length (features per sample): {X.shape[1]}")

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
    elif model_type == 'decision_tree': 
        return DecisionTreeClassifier(
            max_depth=10, 
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'random_forest', 'svm', 'mlp', 'decision_tree'.")

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

print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

observation_models = {}
observation_accuracies = {}
model = None 

if TRAIN_BAGGING:
    print(f"Configured to train a BaggingClassifier with '{BAGGING_BASE_ESTIMATOR}' as base estimator.")
    base_estimator = create_individual_model(BAGGING_BASE_ESTIMATOR)
    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=BAGGING_N_ESTIMATORS,
        max_samples=1.0, 
        max_features=1.0, 
        random_state=42,
        n_jobs=-1, 
        oob_score=True 
    )
    single_base_model_for_comp = create_individual_model(BAGGING_BASE_ESTIMATOR)
    observation_models[f'Single {BAGGING_BASE_ESTIMATOR}'] = single_base_model_for_comp
    MODEL_PATH = "holes_bagging_classifier.pkl" 
elif TRAIN_ENSEMBLE:
    print(f"Configured to train a VotingClassifier (ensemble) with members: {', '.join(ENSEMBLE_MODEL_TYPES)}.")
    estimators = []
    for model_type_name in ENSEMBLE_MODEL_TYPES:
        estimator = create_individual_model(model_type_name)
        estimators.append((model_type_name, estimator))
        observation_models[model_type_name] = estimator 
    
    model = VotingClassifier(
        estimators=estimators, 
        voting='soft', 
        n_jobs=-1 
    )
    MODEL_PATH = "holes_ensemble_classifier.pkl" 
else:
    print("Configured to train a single RandomForestClassifier.")
    model = create_individual_model('random_forest')
    observation_models['random_forest'] = model 
    MODEL_PATH = "holes_random_forest_classifier.pkl" 

if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading scaler: {e}. Re-fitting scaler.")
else:
    print("No pre-existing scaler found. A new scaler will be fitted.")

if os.path.exists(MODEL_PATH):
    print(f"Attempting to load pre-existing model from {MODEL_PATH}...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            loaded_model = pickle.load(f)
            is_bagging_match = TRAIN_BAGGING and isinstance(loaded_model, BaggingClassifier)
            is_ensemble_match = TRAIN_ENSEMBLE and isinstance(loaded_model, VotingClassifier) and not TRAIN_BAGGING 
            is_single_rf_match = not TRAIN_BAGGING and not TRAIN_ENSEMBLE and isinstance(loaded_model, RandomForestClassifier)

            if is_bagging_match or is_ensemble_match or is_single_rf_match:
                model = loaded_model
                print("Model loaded successfully.")
            else:
                print("Loaded model type does not match current configuration. Training from scratch.")
    except Exception as e:
        print(f"Error loading model: {e}. Training from scratch.")
else:
    print("No pre-existing model found for current configuration. Training from scratch.")

# =========================
# 6. TRAINING
# =========================
print(f"\nStarting training...")

if TRAIN_BAGGING:
    print(f"Training single base estimator '{BAGGING_BASE_ESTIMATOR}' for comparison...")
    single_base_model_for_comp = observation_models[f'Single {BAGGING_BASE_ESTIMATOR}']
    single_base_model_for_comp.fit(X_train_scaled, y_train)
    y_pred_single = single_base_model_for_comp.predict(X_test_scaled)
    observation_accuracies[f'Single {BAGGING_BASE_ESTIMATOR}'] = accuracy_score(y_test, y_pred_single)
    print(f"Single {BAGGING_BASE_ESTIMATOR} training completed. Test Accuracy: {observation_accuracies[f'Single {BAGGING_BASE_ESTIMATOR}']:.4f}")

    print(f"Training BaggingClassifier with {BAGGING_N_ESTIMATORS} estimators...")
    model.fit(X_train_scaled, y_train)
    
    if hasattr(model, 'oob_score_'):
        print(f"Bagging Out-of-Bag (OOB) Score: {model.oob_score_:.4f}")

elif TRAIN_ENSEMBLE:
    print("Training individual models for the ensemble...")
    for name, estimator in tqdm(model.estimators, desc="Fitting individual estimators"):
        print(f"  Fitting {name}...")
        estimator.fit(X_train_scaled, y_train)
        y_pred_individual = estimator.predict(X_test_scaled)
        observation_accuracies[name] = accuracy_score(y_test, y_pred_individual)
        print(f"  {name} Test Accuracy: {observation_accuracies[name]:.4f}")
    
    print("Fitting the VotingClassifier (combining individual predictions)...")
    model.fit(X_train_scaled, y_train)
else:
    print(f"Training single model: {model.__class__.__name__}...")
    model.fit(X_train_scaled, y_train)

print("Training completed!")

# =========================
# 7. EVALUATION
# =========================
print("\nEvaluating final model performance...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Final Model ({model.__class__.__name__}) Test Accuracy: {accuracy:.4f}")
print("\nClassification Report for Final Model:")
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
# 9. OBSERVATION / VISUALIZATION
# =========================
print("\n--- Observation and Visualization ---")

if TRAIN_BAGGING:
    labels = [f'Single {BAGGING_BASE_ESTIMATOR}', f'Bagged {BAGGING_BASE_ESTIMATOR} ({BAGGING_N_ESTIMATORS} est.)']
    accuracies = [observation_accuracies[f'Single {BAGGING_BASE_ESTIMATOR}'], accuracy]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, accuracies, color=['skyblue', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: Single vs. Bagged Estimator')
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.ylim(0, 1) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() 
    plt.show()
    print(f"\nObserved that BaggingClassifier with {BAGGING_N_ESTIMATORS} estimators improved accuracy from {accuracies[0]:.4f} to {accuracies[1]:.4f} for the {BAGGING_BASE_ESTIMATOR}.")

elif TRAIN_ENSEMBLE:
    individual_accuracies = list(observation_accuracies.values())
    individual_labels = list(observation_accuracies.keys())
    
    all_labels = individual_labels + ['VotingClassifier (Ensemble)']
    all_accuracies = individual_accuracies + [accuracy]

    plt.figure(figsize=(10, 7))
    colors = ['lightblue'] * len(individual_labels) + ['mediumseagreen']
    plt.bar(all_labels, all_accuracies, color=colors)
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: Individual Ensemble Members vs. VotingClassifier')
    plt.xticks(rotation=45, ha='right') 
    for i, acc in enumerate(all_accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("\nObserved individual model accuracies and the final ensemble accuracy.")
    print("The VotingClassifier often combines the strengths of its base models to achieve better overall performance.")

else:
    print(f"\nOnly a single {model.__class__.__name__} was trained. No ensemble/bagging comparison visualization generated.")

# =========================
# 10. INFERENCE
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
    print(f"Final Predicted Damage Level: {class_labels[predicted_label_idx]}")

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
