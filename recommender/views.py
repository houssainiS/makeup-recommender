from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from PIL import Image
import base64
import io

from recommender.models.ml_model import predict

def home(request):
    return render(request, "recommender/home.html")


@csrf_exempt
def upload_photo(request):
    if request.method == "POST":
        try:
            # Load image from file input or base64
            if 'photo' in request.FILES:
                photo_file = request.FILES['photo']
                image = Image.open(photo_file).convert('RGB')
            else:
                data_url = request.POST.get('photo')
                header, encoded = data_url.split(",", 1)
                decoded = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(decoded)).convert('RGB')

            # Predict using models
            preds = predict(image)

            # Check if predict returned an error key (e.g., no face found)
            if "error" in preds:
                return JsonResponse({"error": preds["error"]}, status=400)

            skin_type = preds['type_pred'].lower()
            skin_defect = preds['defect_pred'].lower()
            recommendation = preds.get("recommendation", "No recommendation available.")

            # Prepare cropped face image as base64 for frontend display
            cropped_face = preds.get("cropped_face")
            buffered = io.BytesIO()
            cropped_face.save(buffered, format="JPEG")
            cropped_face_base64 = base64.b64encode(buffered.getvalue()).decode()

            return JsonResponse({
                "skin_type": skin_type.title(),
                "skin_defect": skin_defect.title(),
                "recommendation": recommendation,
                "cropped_face": f"data:image/jpeg;base64,{cropped_face_base64}"
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)
