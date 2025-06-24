import os
from pathlib import Path
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image # Import Image for predict_image function

# ============ CONFIG ============
# IMPORTANT: Ensure DATA_DIR contains subfolders like 'Normal', 'Mn_Insufficient', etc.
DATA_DIR = "./"
BATCH_SIZE = 32
EPOCHS = 20 # Increased epochs for better multi-class learning, adjust as needed
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============ DATASET ============
torch.manual_seed(SEED)
dataset = datasets.ImageFolder(DATA_DIR, transform=transform_train)

# Get class names and their mapping automatically from ImageFolder
CLASS_NAMES = dataset.classes # This will be something like ['Fe_Insufficient', 'Mg_Insufficient', 'Mn_Insufficient', 'Normal', 'Zn_Insufficient']
NUM_CLASSES = len(CLASS_NAMES)
print(f"Detected Classes: {CLASS_NAMES}")
print(f"Number of Classes: {NUM_CLASSES}")

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

val_ds.dataset.transform = transform_val


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ============ MODEL ============
model = models.resnet18(pretrained=True)
# Adjust the final fully connected layer for NUM_CLASSES
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ============ TRAINING ============
# Change criterion to CrossEntropyLoss for multi-class classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train():
    model.train()
    total_loss = 0
    # Labels for CrossEntropyLoss should be long (integer class IDs), not float
    for imgs, labels in tqdm(train_loader, desc="Training"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE) 

        outputs = model(imgs)
        loss = criterion(outputs, labels) # CrossEntropyLoss expects (N, C) logits and (N) integer labels

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
            # For multi-class, get the index of the max logit as the prediction
            preds = outputs.argmax(dim=1)
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

# ============ PREDICTION FUNCTION ============
def predict_image(path):
    model.eval()
    # Load the state dict before prediction if it wasn't loaded for training
    # or if you are running this function in a separate script/session.
    # model.load_state_dict(torch.load("leaf_classifier.pth", map_location=DEVICE))
    # model.to(DEVICE)

    image = Image.open(path).convert("RGB")
    transform = transform_val # use validation transform for prediction
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        # Apply softmax to get probabilities for each class
        probabilities = torch.softmax(output, dim=1)[0] # Get probabilities for the single image
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()

        predicted_label = CLASS_NAMES[predicted_class_idx]

        print(f"Prediction: {predicted_label} ({confidence*100:.2f}% confidence)")

# ============ Example usage ============
# You need to have an actual image file in one of your new class subdirectories
# for this to work. E.g., 'DATA_DIR/Normal/some_normal_leaf.jpg'
# Or 'DATA_DIR/Mn_Insufficient/some_mn_leaf.jpg'

# Example: assuming you have an image called 'test_leaf.jpg' in your DATA_DIR
# For testing, make sure to put a dummy image for testing, e.g., 'Normal/test_normal.jpg'
# predict_image('Normal/test_normal.jpg') # Replace with an actual path to your test image

# Example of a dummy file path (you need to create this image for testing)
# predict_image(os.path.join(DATA_DIR, 'Normal', 'some_normal_leaf_image.jpg'))
# predict_image(os.path.join(DATA_DIR, 'Mn_Insufficient', 'some_mn_leaf_image.jpg'))

# Placeholder for actual usage, assuming 'test_image.jpg' exists in your root DATA_DIR
# If your images are within the class subfolders, you'd specify that path
# E.g., predict_image(os.path.join(DATA_DIR, 'Normal', 'a_normal_leaf.jpg'))
print("\n--- Example Prediction ---")
# Replace with an actual path to an image file within your dataset structure
# For example, if you have a test image for Healthy leaves:
# predict_image("path/to/your/Normal/healthy_leaf_example.jpg")
# If you have an image for Manganese insufficiency:
# predict_image("path/to/your/Mn_Insufficient/mn_deficient_leaf_example.jpg")
# For a quick test, you might use an image that was part of your training data (not ideal for evaluation, but works for checking functionality)
try:
    # This will attempt to find the first image in the validation set for prediction example
    # This is a bit advanced, just provide a static path if it's easier.
    example_img_path = val_ds.dataset.samples[val_ds.indices[0]][0] # Get path of first image in validation set
    predict_image(example_img_path)
except Exception as e:
    print(f"Could not automatically get an example image path from val_ds: {e}")
    print("Please manually update the predict_image call with a valid image path, e.g., predict_image('path/to/your/Normal/leaf1.jpg')")