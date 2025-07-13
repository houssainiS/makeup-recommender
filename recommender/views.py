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
            if 'photo' in request.FILES:
                photo_file = request.FILES['photo']
                image = Image.open(photo_file).convert('RGB')
            else:
                data_url = request.POST.get('photo')
                header, encoded = data_url.split(",", 1)
                decoded = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(decoded)).convert('RGB')

            preds = predict(image)

            skin_type = preds['type_pred'].lower()
            skin_defect = preds['defect_pred'].lower()

            # Use recommendation from model directly
            recommendation = preds.get("recommendation", "No recommendation available.")

            return JsonResponse({
                "skin_type": skin_type.title(),
                "skin_defect": skin_defect.title(),
                "recommendation": recommendation
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)
