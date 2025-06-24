import torch
from torchvision import transforms, datasets
from PIL import Image
import pickle

DATA_DIR_FOR_CLASS_NAMES = "./data"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pkl"

transform_inference = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model_and_class_names(model_path=MODEL_PATH, data_dir_for_class_names=DATA_DIR_FOR_CLASS_NAMES):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.eval().to(DEVICE)

    dataset = datasets.ImageFolder(data_dir_for_class_names, transform=transforms.ToTensor())
    class_names = dataset.classes

    return model, class_names

def predict_image_data(image_path, model, class_names, device=DEVICE, transform=transform_inference):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)[0]
        idx = torch.argmax(probs).item()
        return {
            "class": class_names[idx],
            "confidence": round(probs[idx].item(), 4)
        }

# print(predict_image_data("example.jpg", *load_model_and_class_names()))