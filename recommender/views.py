from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from PIL import Image
import base64
import io

# ✅ Correct function name
from recommender.models.ml_model import predict

def home(request):
    return render(request, "recommender/home.html")

@csrf_exempt
def upload_photo(request):
    if request.method == "POST":
        try:
            # 1. Load image from file upload or base64
            if 'photo' in request.FILES:
                photo_file = request.FILES['photo']
                image = Image.open(photo_file).convert('RGB')
            else:
                data_url = request.POST.get('photo')
                header, encoded = data_url.split(",", 1)
                decoded = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(decoded)).convert('RGB')

            # 2. Predict using ML model
            preds = predict(image)

            skin_type = preds['type_pred'].lower()
            skin_defect = preds['defect_pred'].lower()

            # 3. Recommendation logic
            makeup_suggestions = {
                ('dry', 'acne'): "Use hydrating, non-comedogenic foundation and gentle acne treatment primers.",
                ('dry', 'redness'): "Use soothing foundation with redness relief, and calming concealer.",
                ('dry', 'bags'): "Use brightening concealer and moisturizing eye creams.",
                ('dry', 'none'): "Use hydrating foundation with natural finish and nourishing primers.",
                ('oily', 'acne'): "Use mattifying, oil-free foundation and salicylic acid primer for acne control.",
                ('oily', 'redness'): "Use oil-free foundation with green color corrector and mattifying primers.",
                ('oily', 'bags'): "Use lightweight concealer and mattifying eye creams.",
                ('oily', 'none'): "Use mattifying foundation and oil-control primers.",
                ('normal', 'acne'): "Use light-coverage foundation and gentle acne primers.",
                ('normal', 'redness'): "Use foundation with redness control and light concealer.",
                ('normal', 'bags'): "Use brightening concealer and moisturizing eye cream.",
                ('normal', 'none'): "Use balanced foundation and lightweight primers.",
            }

            recommendation = makeup_suggestions.get(
                (skin_type, skin_defect),
                "No specific recommendation available for this combination."
            )

            return JsonResponse({
                "skin_type": skin_type.title(),
                "skin_defect": skin_defect.title(),
                "recommendation": recommendation
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)
