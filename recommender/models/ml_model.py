import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from recommender.models.facemesh_model import detect_and_crop_face, crop_left_eye, crop_right_eye

# ----------------------
# 📍 Path Setup
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKIN_TYPE_MODEL_PATH = os.path.join(BASE_DIR, "skin_model.pth")
SKIN_DEFECT_MODEL_PATH = os.path.join(BASE_DIR, "skin_disease_resnet18.pth")
EYE_COLOR_MODEL_PATH = os.path.join(BASE_DIR, "eye_color_model.pth")

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
        return self.conv(x), self.fc(self.conv(x))

class EyeColorResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(EyeColorResNet, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# ----------------------
# 🔁 Image Transforms
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

transform_eye = transform_defect  # Same as defect model (224x224 normalized)

# ----------------------
# ⚙️ Load Models
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

type_model = SkinCNN()
type_model.load_state_dict(torch.load(SKIN_TYPE_MODEL_PATH, map_location=device))
type_model.to(device).eval()

defect_model = models.resnet18(weights=None)
defect_model.fc = nn.Linear(defect_model.fc.in_features, 4)
defect_model.load_state_dict(torch.load(SKIN_DEFECT_MODEL_PATH, map_location=device))
defect_model.to(device).eval()

eye_model = EyeColorResNet(num_classes=6)
eye_model.load_state_dict(torch.load(EYE_COLOR_MODEL_PATH, map_location=device))
eye_model.to(device).eval()

# ----------------------
# 🏷️ Class Labels
# ----------------------
skin_type_labels = ['dry', 'normal', 'oily']
skin_defect_labels = ['acne', 'redness', 'bags', 'none']
eye_color_labels = ['Amber', 'Blue', 'Brown', 'Green', 'Grey', 'Hazel']

# ----------------------
# 🔮 Main Prediction Function
# ----------------------
def predict(image: Image.Image) -> dict:
    image = image.convert("RGB")

    try:
        face_image = detect_and_crop_face(image)
        left_eye = crop_left_eye(image)
        right_eye = crop_right_eye(image)
    except ValueError as e:
        return {"error": str(e)}

    # Prepare inputs
    input_type = transform_type(face_image).unsqueeze(0).to(device)
    input_defect = transform_defect(face_image).unsqueeze(0).to(device)
    input_left_eye = transform_eye(left_eye).unsqueeze(0).to(device)
    input_right_eye = transform_eye(right_eye).unsqueeze(0).to(device)

    with torch.no_grad():
        type_out = type_model(input_type)[1]
        defect_out = defect_model(input_defect)
        left_eye_out = torch.sigmoid(eye_model(input_left_eye)).cpu().numpy()[0]
        right_eye_out = torch.sigmoid(eye_model(input_right_eye)).cpu().numpy()[0]

        # Classifications
        type_probs = F.softmax(type_out, dim=1).cpu().numpy()[0]
        defect_probs = F.softmax(defect_out, dim=1).cpu().numpy()[0]
        type_pred = skin_type_labels[int(type_probs.argmax())]
        defect_pred = skin_defect_labels[int(defect_probs.argmax())]

        # Find top eye color strings
        left_eye_dict = dict(zip(eye_color_labels, [float(p) for p in left_eye_out]))
        right_eye_dict = dict(zip(eye_color_labels, [float(p) for p in right_eye_out]))
        left_eye_color = max(left_eye_dict, key=left_eye_dict.get)
        right_eye_color = max(right_eye_dict, key=right_eye_dict.get)

    return {
        "type_pred": type_pred,
        "type_probs": type_probs.tolist(),
        "defect_pred": defect_pred,
        "defect_probs": defect_probs.tolist(),
        # Send only the top eye color strings, not full dicts
        "left_eye_color": left_eye_color,
        "right_eye_color": right_eye_color,
        "cropped_face": face_image,
    }

