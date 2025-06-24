import os
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch
import pickle

# ============ CONFIG ============
# Adjust this path to your test data directory
TEST_DIR = "./test"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pkl" # Make sure this matches where you saved your model

# ============ DATA TRANSFORM ============
# Use the same validation transform as during training
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============ LOAD MODEL AND CLASS NAMES ============
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    model.eval() # Set model to evaluation mode
    model = model.to(DEVICE)
    print(f"Model loaded successfully from {MODEL_PATH}")

    from torchvision import datasets # Need this for ImageFolder
    
    try:
 
        temp_dataset = datasets.ImageFolder("./data", transform=transforms.ToTensor())
        CLASS_NAMES = temp_dataset.classes
        print(f"Class names loaded: {CLASS_NAMES}")
    except Exception as e:
        print(f"Warning: Could not automatically determine CLASS_NAMES from './data'. "
              f"Please ensure './data' exists with original class structure or manually define CLASS_NAMES. Error: {e}")
        # Fallback: You MUST replace this with your actual class names if the above fails
        # Example: CLASS_NAMES = ['ClassA', 'ClassB', 'ClassC']
        CLASS_NAMES = [] # Placeholder if you can't load them dynamically


except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Please ensure the model is trained and saved.")
    exit()
except Exception as e:
    print(f"Error loading model or getting class names: {e}")
    exit()

# ============ PREDICTION FUNCTION ============
def predict_image(image_path, model, transform, class_names, device):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()

            predicted_label = class_names[predicted_class_idx]
            return predicted_label, confidence
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return "ERROR", 0.0

# ============ PREDICT ON TEST FOLDERS ============
print(f"\n--- Predicting on images in '{TEST_DIR}' ---")
if not os.path.isdir(TEST_DIR):
    print(f"Error: Test directory '{TEST_DIR}' not found. Please create it and place image folders inside.")
else:
    for class_folder in sorted(os.listdir(TEST_DIR)):
        class_folder_path = os.path.join(TEST_DIR, class_folder)
        if os.path.isdir(class_folder_path):
            print(f"\nFolder: {class_folder} (True Label)")
            
            images_in_folder = [f for f in os.listdir(class_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            if not images_in_folder:
                print(f"  No images found in {class_folder_path}")
                continue

            for image_name in images_in_folder:
                image_path = os.path.join(class_folder_path, image_name)
                predicted_label, confidence = predict_image(image_path, model, transform_test, CLASS_NAMES, DEVICE)
                print(f"  Image: {image_name:<20} | Predicted: {predicted_label:<15} | Confidence: {confidence*100:.2f}%")