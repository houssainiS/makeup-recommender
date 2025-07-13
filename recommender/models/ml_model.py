import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ----------------------
# 📍 Path Setup
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKIN_TYPE_MODEL_PATH = os.path.join(BASE_DIR, "skin_model.pth")
SKIN_DEFECT_MODEL_PATH = os.path.join(BASE_DIR, "skin_disease_resnet18.pth")

# ----------------------
# 🧠 Model Definitions
# ----------------------
class SkinCNN(nn.Module):
    def __init__(self):
        super(SkinCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 30 * 30, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ----------------------
# 🔁 Transforms
# ----------------------
transform_type = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

transform_defect = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------
# 🧠 Load Models
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Skin type model (3 classes)
type_model = SkinCNN()
type_model.load_state_dict(torch.load(SKIN_TYPE_MODEL_PATH, map_location=device))
type_model.to(device)
type_model.eval()

# Skin defect model (4 classes)
defect_model = models.resnet18(weights=None)
defect_model.fc = nn.Linear(defect_model.fc.in_features, 4)
defect_model.load_state_dict(torch.load(SKIN_DEFECT_MODEL_PATH, map_location=device))
defect_model.to(device)
defect_model.eval()

# ----------------------
# 🧪 Prediction Function
# ----------------------
skin_type_labels = ['dry', 'normal', 'oily']
skin_defect_labels = ['acne', 'redness', 'bags', 'none']

def predict(image: Image.Image) -> dict:
    # Ensure RGB format
    image = image.convert("RGB")

    # Apply transforms
    input_type = transform_type(image).unsqueeze(0).to(device)
    input_defect = transform_defect(image).unsqueeze(0).to(device)

    with torch.no_grad():
        type_out = type_model(input_type)
        defect_out = defect_model(input_defect)

        type_probs = F.softmax(type_out, dim=1).cpu().numpy()[0]
        defect_probs = F.softmax(defect_out, dim=1).cpu().numpy()[0]

        type_pred = skin_type_labels[int(type_probs.argmax())]
        defect_pred = skin_defect_labels[int(defect_probs.argmax())]

    return {
        "type_pred": type_pred,
        "defect_pred": defect_pred,
        "type_probs": type_probs.tolist(),
        "defect_probs": defect_probs.tolist(),
    }
