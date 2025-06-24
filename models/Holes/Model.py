import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# =========================
# 1. CONFIG
# =========================
BASE_DIR = "./data"
FOLDERS = ["Healthy", "Light Damage", "Medium Damage", "Severe Damage"]
CLASS_LABELS = ["Healthy", "Light", "Medium", "Severe"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
WEIGHTS_PATH = "signal_classifier.pth" # Added for model weights

# =========================
# 2. UTILS
# =========================
def read_signal_file(filepath):
    df = pd.read_csv(filepath, sep="\t", encoding="latin1", names=["Frequency", "Magnitude", "Phase"], header=0)
    return df["Magnitude"].values.astype(np.float32)

# =========================
# 3. DATASET
# =========================
class SignalDataset(Dataset):
    def __init__(self, base_dir, folders):
        self.samples = []
        self.labels = []
        for label, folder in enumerate(folders):
            folder_path = os.path.join(base_dir, folder)
            for fname in os.listdir(folder_path):
                if fname.endswith(".txt"):
                    path = os.path.join(folder_path, fname)
                    try:
                        mag = read_signal_file(path)
                        self.samples.append(mag)
                        self.labels.append(label)
                    except:
                        continue

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# =========================
# 4. MODEL
# =========================
class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Calculate the flattened size dynamically
        # Create a dummy input to pass through the conv layers
        # The input shape is (batch_size, channels, length)
        # Use 1 for batch_size as we only care about the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size)
            flattened_size = self.conv_layers(dummy_input).view(1, -1).shape[1]

        self.linear_layers = nn.Sequential(
            nn.Flatten(), # Flatten here after the dummy pass
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x) # Flattening is now part of linear_layers
        return x

# =========================
# 5. TRAINING
# =========================
all_data = SignalDataset(BASE_DIR, FOLDERS)
input_size = all_data[0][0].shape[0]

X_train, X_test, y_train, y_test = train_test_split(
    all_data.samples, all_data.labels, test_size=0.2, random_state=42, stratify=all_data.labels
)

X_train = torch.tensor(X_train).unsqueeze(1).to(DEVICE)
# Change y_train to torch.long
y_train = torch.tensor(y_train).long().to(DEVICE)
X_test = torch.tensor(X_test).unsqueeze(1).to(DEVICE)
# Change y_test to torch.long
y_test = torch.tensor(y_test).long().to(DEVICE)

train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

model = CNN1D(input_size, len(FOLDERS)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Load model weights if they exist
if os.path.exists(WEIGHTS_PATH):
    print(f"Loading model weights from {WEIGHTS_PATH}")
    # Ensure map_location is set correctly for your device
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
else:
    print("No pre-existing model weights found. Training from scratch.")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# =========================
# 6. EVALUATION
# =========================
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        pred_classes = preds.argmax(dim=1)
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(pred_classes.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

# =========================
# 7. SAVE MODEL
# =========================
print(f"Saving model weights to {WEIGHTS_PATH}")
torch.save(model.state_dict(), WEIGHTS_PATH)

# =========================
# 8. INFERENCE
# =========================
def predict_observation(file_path):
    model.eval()
    mag = read_signal_file(file_path)
    mag = torch.tensor(mag).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(mag)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        for i, p in enumerate(probs):
            print(f"{CLASS_LABELS[i]}: {p*100:.2f}%")
        print("Prediction:", CLASS_LABELS[np.argmax(probs)])

# Example usage:
# Make sure "Observation.txt" exists in your current directory for this to work
# You can create a dummy file for testing:
# with open("Observation.txt", "w") as f:
#     f.write("Frequency\tMagnitude\tPhase\n")
#     for i in range(100):
#         f.write(f"{i}\t{np.random.rand()}\t{np.random.rand()}\n")
predict_observation("Observation.txt")