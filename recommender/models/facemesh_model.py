from PIL import Image
from facenet_pytorch import MTCNN
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

def detect_and_crop_face(image: Image.Image) -> Image.Image:
    """
    Detects face using MTCNN and crops image to face region.
    Raises ValueError if no face or multiple faces detected.
    """
    boxes, _ = mtcnn.detect(image)

    if boxes is None or len(boxes) != 1:
        raise ValueError("Please upload a clear photo with exactly one face.")

    box = boxes[0]
    x1, y1, x2, y2 = [int(b) for b in box]
    face = image.crop((x1, y1, x2, y2))

    return face
