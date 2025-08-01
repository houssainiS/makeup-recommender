import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 Model Once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)

def detect_skin_defects_yolo(image: Image.Image, conf_threshold=0.3):
    """
    Detect skin defects using YOLOv8.
    Returns:
      - detections: list of dicts with 'bbox', 'label', 'confidence'
      - annotated_image: PIL Image with boxes drawn
    """
    # Convert PIL to OpenCV
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Predict
    results = yolo_model.predict(source=image_cv2, conf=conf_threshold, stream=False)[0]

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            # bbox in xyxy format (float), convert to int list
            bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()  # [x1, y1, x2, y2]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo_model.names[cls_id]

            detections.append({
                "bbox": bbox,
                "label": label,
                "confidence": conf
            })

            # Draw box on image
            cv2.rectangle(image_cv2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image_cv2, f"{label} {conf:.2f}", (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert annotated image back to PIL
    annotated_image = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

    return detections, annotated_image
