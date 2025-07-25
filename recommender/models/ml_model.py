import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Import the face detection/cropping function from your new file
from recommender.models.facemesh_model import detect_and_crop_face

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
    """Custom CNN for skin type classification (3 classes)."""
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

# ----------------------
# ⚙️ Load Models
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load skin type model
type_model = SkinCNN()
type_model.load_state_dict(torch.load(SKIN_TYPE_MODEL_PATH, map_location=device))
type_model.to(device)
type_model.eval()

# Load skin defect model (ResNet18)
defect_model = models.resnet18(weights=None)
defect_model.fc = nn.Linear(defect_model.fc.in_features, 4)
defect_model.load_state_dict(torch.load(SKIN_DEFECT_MODEL_PATH, map_location=device))
defect_model.to(device)
defect_model.eval()

# ----------------------
# 🏷️ Class Labels
# ----------------------
skin_type_labels = ['dry', 'normal', 'oily']
skin_defect_labels = ['acne', 'redness', 'bags', 'none']

# ----------------------
# 🎨 Makeup Recommendation Logic
# ----------------------
def advanced_makeup_recommendation(defect_probs, type_probs, defect_labels, type_labels):
    recommendation = ""
    main_defect_idx = int(defect_probs.argmax())
    main_type_idx = int(type_probs.argmax())

    main_defect = defect_labels[main_defect_idx]
    main_type = type_labels[main_type_idx]

    # Analyze skin defect
    if defect_probs[main_defect_idx] >= 0.7:
        recommendation += f"🩺 Main issue: **{main_defect.upper()}** (confident).\n"
        if main_defect == "acne":
            recommendation += "👉 Use full acne-control routine with salicylic acid base and matte foundation.\n"
        elif main_defect == "redness":
            recommendation += "👉 Apply calming primer and green-tinted concealer.\n"
        elif main_defect == "bags":
            recommendation += "👉 Use brightening concealer and cold eye gel before makeup.\n"
    elif defect_probs[-1] >= 0.6:
        recommendation += "🩺 Mostly clear skin (None > 60%).\n"
        recommendation += "👉 Recommend lightweight, natural-look makeup with glow finish.\n"
    else:
        minor = [(defect_labels[i], p) for i, p in enumerate(defect_probs)
                 if 0.2 < p < 0.6 and defect_labels[i] != "none"]
        if minor:
            issues_text = ", ".join([f"{d} ({p:.0%})" for d, p in minor])
            recommendation += f"🩺 Minor skin concerns: {issues_text}\n"
            recommendation += "👉 Use light corrector only on affected areas.\n"
        else:
            recommendation += "🩺 No strong issues detected. Use clean base makeup.\n"

    # Analyze skin type
    if type_probs[main_type_idx] >= 0.75:
        recommendation += f"💧 Dominant skin type: **{main_type.upper()}**.\n"
        if main_type == "oily":
            recommendation += "👉 Use mattifying primer, oil-free foundation, and powder finish.\n"
        elif main_type == "dry":
            recommendation += "👉 Use hydrating foundation, creamy concealer, and avoid powders.\n"
        elif main_type == "normal":
            recommendation += "👉 Use balanced, natural-look foundation and light primers.\n"
    elif max(type_probs) - min(type_probs) < 0.25:
        recommendation += "💧 Skin appears **combination or mixed**.\n"
        recommendation += "👉 Suggest dual-zone skincare or adaptive foundation (e.g., matte T-zone, hydrating elsewhere).\n"
    else:
        recommendation += f"💧 Slightly leaning towards **{main_type}**, but mixed.\n"
        recommendation += "👉 Use adaptable formulas (e.g., semi-matte or hydrating matte).\n"

    recommendation += "\n💄 **Overall Suggestion**: Focus makeup only where needed, use breathable layers, and customize by zone."

    return recommendation

# ----------------------
# 🔮 Main Prediction Function
# ----------------------
def predict(image: Image.Image) -> dict:
    """
    Full pipeline:
    1. Detect and crop face.
    2. Predict skin type and defect.
    3. Generate makeup recommendation.
    4. Return predictions and cropped face or error.
    """
    image = image.convert("RGB")

    try:
        # 🔍 Step 1: Detect & crop face using the imported function
        face_image = detect_and_crop_face(image)
    except ValueError as e:
        # Return an error key in the dict so your view can handle it
        return {"error": str(e)}

    # 🔄 Step 2: Apply transforms and predict
    input_type = transform_type(face_image).unsqueeze(0).to(device)
    input_defect = transform_defect(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        type_out = type_model(input_type)
        defect_out = defect_model(input_defect)

        type_probs = F.softmax(type_out, dim=1).cpu().numpy()[0]
        defect_probs = F.softmax(defect_out, dim=1).cpu().numpy()[0]

        type_pred = skin_type_labels[int(type_probs.argmax())]
        defect_pred = skin_defect_labels[int(defect_probs.argmax())]

        recommendation = advanced_makeup_recommendation(defect_probs, type_probs,
                                                        skin_defect_labels, skin_type_labels)

    # 🔚 Step 3: Return everything (cropped face image included)
    return {
        "type_pred": type_pred,
        "defect_pred": defect_pred,
        "type_probs": type_probs.tolist(),
        "defect_probs": defect_probs.tolist(),
        "recommendation": recommendation,
        "cropped_face": face_image
    }
