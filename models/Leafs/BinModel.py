import os
from pathlib import Path
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ============ CONFIG ============
DATA_DIR = "./"  # contains /Normal and /Insufficient
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
VAL_SPLIT = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ AUGMENTATION ============
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
])

# ============ DATASET ============
torch.manual_seed(SEED)
dataset = datasets.ImageFolder(DATA_DIR, transform=transform_train)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

val_ds.dataset.transform = transform_val

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ============ MODEL ============
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary output
model = model.to(DEVICE)

# ============ TRAINING ============
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train():
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc="Training"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
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
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(imgs)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ============ MAIN LOOP ============
for epoch in range(EPOCHS):
    loss = train()
    acc = evaluate()
    print(f"[Epoch {epoch+1}] Loss: {loss:.4f} | Val Accuracy: {acc*100:.2f}%")

# Save final model
torch.save(model.state_dict(), "leaf_classifier.pth")
print("Model saved to leaf_classifier.pth")

def predict_image(path):
    model.eval()
    image = Image.open(path).convert("RGB")
    transform = transform_val  # use validation transform
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output).item()

        label_map = {0: "Insufficient", 1: "Normal"}  # Adjust if needed
        label = label_map[int(pred > 0.5)]
        confidence = pred if pred > 0.5 else 1 - pred

        print(f"Prediction: {label} ({confidence*100:.2f}% confidence)")

# ============ Example usage ============

predict_image('Healthy_003.jpg')