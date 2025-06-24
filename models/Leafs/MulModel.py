import os
from pathlib import Path
from torchvision import transforms, datasets, models
from torchvision.models import ResNet18_Weights 
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import pickle

DATA_DIR = "./data"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 224
VAL_SPLIT = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

torch.manual_seed(SEED)
dataset = datasets.ImageFolder(DATA_DIR, transform=transform_train)

CLASS_NAMES = dataset.classes
NUM_CLASSES = len(CLASS_NAMES)
print(f"Detected Classes: {CLASS_NAMES}")
print(f"Number of Classes: {NUM_CLASSES}")

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

val_ds.dataset.transform = transform_val

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train():
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc="Training"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE) 

        outputs = model(imgs)
        loss = criterion(outputs, labels) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE) 

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

for epoch in range(EPOCHS):
    loss = train()
    acc = evaluate()
    print(f"[Epoch {epoch+1}] Loss: {loss:.4f} | Val Accuracy: {acc*100:.2f}%")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to model.pkl")

def predict_image(path):
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    loaded_model.eval()
    loaded_model = loaded_model.to(DEVICE)

    image = Image.open(path).convert("RGB")
    transform = transform_val 
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = loaded_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0] 
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()

        predicted_label = CLASS_NAMES[predicted_class_idx]

        print(f"Prediction: {predicted_label} ({confidence*100:.2f}% confidence)")

print("\n--- Example Prediction ---")
try:
    example_img_path = val_ds.dataset.samples[val_ds.indices[0]][0]
    predict_image(example_img_path)
except Exception as e:
    print(f"Could not automatically get an example image path from val_ds: {e}")
    print("Please manually update the predict_image call with a valid image path, e.g., predict_image('path/to/your/Normal/leaf1.jpg')")