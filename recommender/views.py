from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from PIL import Image
import base64
import io

from recommender.models.ml_model import predict
from recommender.models.yolo_model import detect_skin_defects_yolo


def home(request):
    return render(request, "recommender/home.html")


@csrf_exempt
def upload_photo(request):
    if request.method == "POST":
        try:
            # Load image from uploaded file or base64 string
            if 'photo' in request.FILES:
                photo_file = request.FILES['photo']
                image = Image.open(photo_file).convert('RGB')
            else:
                data_url = request.POST.get('photo')
                header, encoded = data_url.split(",", 1)
                decoded = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(decoded)).convert('RGB')

            # Run main classifier (skin type + skin defect + eyes)
            preds = predict(image)
            if "error" in preds:
                return JsonResponse({"error": preds["error"]}, status=400)

            skin_type = preds['type_pred'].lower()
            skin_defect = preds['defect_pred'].lower()
            cropped_face = preds.get("cropped_face")

            # Convert cropped face to base64
            buffered = io.BytesIO()
            cropped_face.save(buffered, format="JPEG")
            cropped_face_base64 = base64.b64encode(buffered.getvalue()).decode()

            # Eye colors (top predictions)
            left_eye_color = preds.get("left_eye_color", "Unknown")
            right_eye_color = preds.get("right_eye_color", "Unknown")

            # Run YOLOv8 on cropped face to detect detailed skin defects
            yolo_boxes, yolo_annotated_image = detect_skin_defects_yolo(cropped_face)

            # Convert YOLO annotated image to base64
            buffered_annot = io.BytesIO()
            yolo_annotated_image.save(buffered_annot, format="JPEG")
            yolo_annotated_base64 = base64.b64encode(buffered_annot.getvalue()).decode()

            return JsonResponse({
                "skin_type": skin_type.title(),
                "skin_defect": skin_defect.title(),
                "cropped_face": f"data:image/jpeg;base64,{cropped_face_base64}",
                "type_probs": preds.get("type_probs", []),
                "defect_probs": preds.get("defect_probs", []),
                "yolo_boxes": yolo_boxes,
                "yolo_annotated": f"data:image/jpeg;base64,{yolo_annotated_base64}",
                "left_eye_color": left_eye_color.title(),
                "right_eye_color": right_eye_color.title()
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)
