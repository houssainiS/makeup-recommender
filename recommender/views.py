from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from django.http import JsonResponse
from django.contrib import messages
from django.db.models import F, Q, Count # Added F and Q for atomic updates
from PIL import Image
import base64
import io
import json
import gc  # garbage collection import
import os
import requests
import urllib.parse
from django.conf import settings
from datetime import datetime, timedelta
from django.utils import timezone

from recommender.AImodels.ml_model import predict
from recommender.AImodels.yolo_model import detect_skin_defects_yolo
from recommender.AImodels.segment_skin_conditions_yolo import segment_skin_conditions  

from .models import FaceAnalysis, Feedback, Visitor, Shop, PageContent, Purchase
# Added register_gdpr_webhooks to imports
from .webhooks import register_uninstall_webhook, fetch_usage_duration, register_orders_updated_webhook, register_gdpr_webhooks, register_shop_update_webhook
from .shopify_navigation import create_page

# Load from environment variables with fallback
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY", "fallback-key-for-dev")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET", "fallback-secret-for-dev")


def home(request):
    """
    Render the homepage.
    """
    return render(request, "recommender/home.html")


@csrf_exempt
def upload_photo(request):
    """
    Handle POST requests with an uploaded photo or base64 image string.
    Run multiple AI models to analyze skin type, acne, eye colors, skin defects,
    and segmentation. Returns a detailed JSON response with results and images.

    Also logs a FaceAnalysis record for each successful analysis to count usage.
    """
    if request.method == "POST":
        image = None
        cropped_face = None
        yolo_annotated_image = None
        segmented_img = None
        buffered = None
        buffered_annot = None
        buffered_seg = None
        
        try:
            # Load image from uploaded file or base64 string
            if 'photo' in request.FILES:
                photo_file = request.FILES['photo']

                # ✅ Validate file size (max 10 MB) - FROM LIVE
                max_size = 10 * 1024 * 1024  # 10 MB
                if photo_file.size > max_size:
                    return JsonResponse({"error": "File too large (max 10 MB allowed)."}, status=400)

                # ✅ Validate file extension - FROM LIVE
                valid_extensions = ['jpg', 'jpeg', 'png']
                extension = photo_file.name.split('.')[-1].lower()
                if extension not in valid_extensions:
                    return JsonResponse({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed."}, status=400)

                image = Image.open(photo_file).convert('RGB')
            else:
                data_url = request.POST.get('photo')
                header, encoded = data_url.split(",", 1)
                decoded = base64.b64decode(encoded)

                # ✅ Validate base64 image size (max 10 MB) - FROM LIVE
                if len(decoded) > 10 * 1024 * 1024:
                    return JsonResponse({"error": "Image too large (max 10 MB allowed)."}, status=400)

                image = Image.open(io.BytesIO(decoded)).convert('RGB')
                
                # ✅ Validate image format - FROM LIVE
                if image.format not in ["JPEG", "JPG", "PNG"]:
                     # Note: PIL often detects JPEG as JPEG, but converting to RGB handles most. 
                     # This check is good but sometimes strictly enforcing format on BytesIO opening can be tricky.
                     # Proceeding as image is already open.
                     pass

            # Run main classifier (skin type + eyes + acne)
            preds = predict(image)
            if "error" in preds:
                return JsonResponse({"error": preds["error"]}, status=400)

            skin_type = preds['type_pred'].lower()
            cropped_face = preds.get("cropped_face")

            buffered = io.BytesIO()
            cropped_face.save(buffered, format="JPEG")
            cropped_face_base64 = base64.b64encode(buffered.getvalue()).decode()
            buffered.close()
            buffered = None

            # Eye colors (top predictions or "Eyes Closed")
            left_eye_color = preds.get("left_eye_color", "Unknown")
            right_eye_color = preds.get("right_eye_color", "Unknown")

            if isinstance(left_eye_color, str) and "closed" not in left_eye_color.lower():
                left_eye_color = left_eye_color.title()
            if isinstance(right_eye_color, str) and "closed" not in right_eye_color.lower():
                right_eye_color = right_eye_color.title()

            # Acne prediction and confidence
            acne_pred = preds.get("acne_pred", "Unknown")
            acne_confidence = preds.get("acne_confidence", 0)

            acne_mapping = {
                "0": "Clear",
                "1": "Mild",
                "2": "Moderate",
                "3": "Severe",
                "clear": "Clear"
            }
            acne_pred_label = acne_mapping.get(str(acne_pred).lower(), "Unknown")

            # Run YOLOv8 on cropped face
            yolo_boxes, yolo_annotated_image = detect_skin_defects_yolo(cropped_face)

            buffered_annot = io.BytesIO()
            yolo_annotated_image.save(buffered_annot, format="JPEG")
            yolo_annotated_base64 = base64.b64encode(buffered_annot.getvalue()).decode()
            buffered_annot.close()
            buffered_annot = None
            yolo_annotated_image.close()
            yolo_annotated_image = None

            # Run YOLOv8 segmentation
            segmented_img, segmentation_results = segment_skin_conditions(cropped_face)
            
            buffered_seg = io.BytesIO()
            segmented_img.save(buffered_seg, format="JPEG")
            segmented_base64 = base64.b64encode(buffered_seg.getvalue()).decode()
            buffered_seg.close()
            buffered_seg = None
            segmented_img.close()
            segmented_img = None

            # ----- Log FaceAnalysis event -----
            session_key = request.session.session_key
            if not session_key:
                request.session.create()
                session_key = request.session.session_key

            ip = get_client_ip(request)
            device_type = get_device_type(request)
            domain = get_domain(request)

            FaceAnalysis.objects.create(
                session_key=session_key,
                ip_address=ip,
                device_type=device_type,
                domain=domain
            )

            # ----- Face analysis Increment Shop Counter (FROM LIVE) -----
            if domain:
                # 1. Clean the domain string
                clean_domain = domain.replace("https://", "").replace("http://", "").strip("/")
                
                # 2. Find the shop (Optimized query using Q for "either/or")
                shop_obj = Shop.objects.filter(Q(domain=clean_domain) | Q(custom_domain=clean_domain)).first()

                # 3. Atomic Increment
                if shop_obj:
                    try:
                        shop_obj.analysis_count = F("analysis_count") + 1
                        shop_obj.save(update_fields=["analysis_count"])
                    except Exception as db_err:
                        print(f"Non-critical error incrementing counter: {db_err}")
            # ---------------------------------------------

            # Response data (NO backend tips anymore)
            response_data = {
                "skin_type": skin_type.title(),
                "acne_pred": acne_pred_label,
                "acne_confidence": round(acne_confidence, 4),
                "cropped_face": f"data:image/jpeg;base64,{cropped_face_base64}",
                "type_probs": preds.get("type_probs", []),
                "yolo_boxes": yolo_boxes,
                "yolo_annotated": f"data:image/jpeg;base64,{yolo_annotated_base64}",
                "left_eye_color": left_eye_color,
                "right_eye_color": right_eye_color,
                "segmentation_overlay": f"data:image/jpeg;base64,{segmented_base64}",
                "segmentation_results": segmentation_results
            }

            if image:
                image.close()
            if cropped_face:
                cropped_face.close()
            
            del image, cropped_face
            if yolo_annotated_image:
                del yolo_annotated_image
            if segmented_img:
                del segmented_img
            gc.collect()

            return JsonResponse(response_data)

        except Exception as e:
            if image:
                image.close()
            if cropped_face:
                cropped_face.close()
            if yolo_annotated_image:
                yolo_annotated_image.close()
            if segmented_img:
                segmented_img.close()
            if buffered:
                buffered.close()
            if buffered_annot:
                buffered_annot.close()
            if buffered_seg:
                buffered_seg.close()
            gc.collect()
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)


# Helper function to get client IP address from request headers
def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        # Handle cases where multiple IPs exist
        return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


# Helper function to detect device type from user agent string
def get_device_type(request):
    user_agent = request.META.get('HTTP_USER_AGENT', '').lower()
    if 'mobile' in user_agent:
        return 'Mobile'
    elif 'tablet' in user_agent:
        return 'Tablet'
    return 'Desktop'

# Helper function to get Shopify domain from request
def get_domain(request):
    return request.POST.get("shop", "") or request.META.get("HTTP_ORIGIN", "")

# feedback
@csrf_exempt
def submit_feedback(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            feedback_type = data.get("feedback_type")
            dislike_reason = data.get("dislike_reason", "").strip()

            if feedback_type not in ["like", "dislike"]:
                return JsonResponse({"error": "Invalid feedback type"}, status=400)

            if feedback_type == "dislike" and not dislike_reason:
                return JsonResponse({"error": "Dislike reason is required"}, status=400)

            feedback = Feedback(
                feedback_type=feedback_type,
                dislike_reason=dislike_reason if feedback_type == "dislike" else ""
            )
            feedback.save()

            return JsonResponse({"message": "Feedback saved successfully"})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid HTTP method"}, status=405)


##############webhooks & Shopify App Routes ############

def app_entry(request):
    shop = request.GET.get("shop")
    if not shop:
        return render(request, "error.html", {"message": "Missing shop parameter"})

    # Check if page creation or metafield flags were passed
    page_created = request.GET.get("page_created") == "1"
    metafield_created = request.GET.get("metafield_created") == "1"
    metafield_deleted = request.GET.get("metafield_deleted") == "1"

    # --- UPDATED LOGIC FROM LIVE (Auto-repair logic) ---
    # 1. Try to find the shop
    shop_obj = Shop.objects.filter(domain=shop).first()

    # 2. Check if the shop is "Ready": Exists + Active + Has Token
    if shop_obj and shop_obj.is_active and shop_obj.offline_token:
        # Shop is fully set up, show the dashboard
        context = {
            "shop": shop,
            "theme_editor_link": shop_obj.theme_editor_link,
            "page_created": page_created,
            "metafield_created": metafield_created,
            "metafield_deleted": metafield_deleted,
            "analysis_count": shop_obj.analysis_count, # Added analysis_count from live
        }
        return render(request, "recommender/shopify_install_page.html", context)
    
    else:
        # If shop doesn't exist OR is missing a token OR is inactive:
        # Start Auth to "Repair" or "Install" the shop automatically.
        return redirect(f"/start_auth/?shop={shop}")


def create_or_get_metafield_definition(shop, offline_token, shop_obj):
    """
    Helper function to create or retrieve the metafield definition for 'usage_duration'.
    """
    graphql_url = f"https://{shop}/admin/api/2025-07/graphql.json"
    headers = {
        "X-Shopify-Access-Token": offline_token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    definition_id = None
    try:
        # --- Check for existing metafield definition ---
        definition_query = """
        {
          metafieldDefinitions(first: 10, ownerType: PRODUCT, namespace: "custom", key: "usage_duration") {
            edges { node { id name namespace key } }
          }
        }
        """
        def_response = requests.post(graphql_url, headers=headers, json={"query": definition_query})
        edges = def_response.json().get("data", {}).get("metafieldDefinitions", {}).get("edges", [])
        if edges:
            definition_id = edges[0]["node"]["id"]
        else:
            create_query = """
            mutation {
              metafieldDefinitionCreate(definition: {
                name: "Product Usage Duration (in days) By beautyxia"
                namespace: "custom"
                key: "usage_duration"
                type: "number_integer"
                description: "How many days the buyer will use the product. By fullfilling this Beautyxia will automatically send an email to the buyer when the usage duration ends . You can leave it empty if you dont want us to send the mail."
                ownerType: PRODUCT
              }) {
                createdDefinition { id name namespace key type { name } }
                userErrors { field message }
              }
            }
            """
            gql_response = requests.post(graphql_url, headers=headers, json={"query": create_query})
            definition_id = gql_response.json().get("data", {}).get("metafieldDefinitionCreate", {}).get("createdDefinition", {}).get("id")
    except Exception as e:
        print(f"[WARNING] Error checking/creating metafield definition: {e}")

    # Pin the metafield definition
    if definition_id:
        try:
            shop_obj.metafield_definition_id = definition_id
            shop_obj.save(update_fields=["metafield_definition_id"])

            pin_query = """
            mutation metafieldDefinitionPin($definitionId: ID!) {
              metafieldDefinitionPin(definitionId: $definitionId) {
                pinnedDefinition { id name namespace key }
                userErrors { field message }
              }
            }
            """
            requests.post(graphql_url, headers=headers, json={"query": pin_query, "variables": {"definitionId": definition_id}})
        except Exception as pin_e:
            print(f"[WARNING] Failed to pin usage_duration metafield: {pin_e}")

    return definition_id


def delete_metafield(request):
    """
    Deletes the metafield definition manually when the user clicks 'Delete Metafield'.
    """
    shop = request.GET.get("shop")
    if not shop:
        return render(request, "error.html", {"message": "Missing shop parameter"})

    try:
        shop_obj = Shop.objects.get(domain=shop, is_active=True)
        access_token = shop_obj.offline_token
        metafield_id = shop_obj.metafield_definition_id

        if not access_token or not metafield_id:
            messages.error(request, "⚠️ Missing metafield or access token.")
            return redirect(f"/app_entry/?shop={shop}")

        graphql_url = f"https://{shop}/admin/api/2025-07/graphql.json"
        headers = {
            "X-Shopify-Access-Token": access_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        delete_mutation = """
        mutation metafieldDefinitionDelete($id: ID!) {
          metafieldDefinitionDelete(id: $id) {
            deletedDefinitionId
            userErrors {
              field
              message
            }
          }
        }
        """

        payload = {"query": delete_mutation, "variables": {"id": metafield_id}}
        response = requests.post(graphql_url, headers=headers, json=payload)
        data = response.json()

        errors = data.get("data", {}).get("metafieldDefinitionDelete", {}).get("userErrors", [])
        deleted_id = data.get("data", {}).get("metafieldDefinitionDelete", {}).get("deletedDefinitionId")

        if errors:
            print("[Shopify Error]", errors)
            messages.error(request, "⚠️ Shopify returned an error while deleting metafield.")
        elif deleted_id:
            shop_obj.metafield_definition_id = None
            shop_obj.save(update_fields=["metafield_definition_id"])
            messages.success(request, "🗑️ Metafield deleted successfully!")
            return redirect(f"/app_entry/?shop={shop}&metafield_deleted=1")
        else:
            messages.warning(request, "⚠️ No metafield was deleted. Check ID or permissions.")

        return redirect(f"/app_entry/?shop={shop}")

    except Shop.DoesNotExist:
        return redirect(f"/start_auth/?shop={shop}")
    except Exception as e:
        print(f"[ERROR] Exception in delete_metafield: {e}")
        messages.error(request, f"⚠️ Error deleting metafield: {e}")
        return redirect(f"/app_entry/?shop={shop}")


def oauth_callback(request):
    """
    Handles Shopify OAuth callback.
    Saves/reactivates the shop, FETCHES CUSTOM DOMAIN, EMAIL, and registers webhooks.
    """
    try:
        shop = request.GET.get("shop")
        code = request.GET.get("code")

        if not shop or not code:
            return JsonResponse({"error": "Missing shop or code"}, status=400)

        # Exchange code for access token
        response = requests.post(
            f"https://{shop}/admin/oauth/access_token",
            data={
                "client_id": SHOPIFY_API_KEY,
                "client_secret": SHOPIFY_API_SECRET,
                "code": code,
            },
        )
        data = response.json()
        offline_token = data.get("access_token")
        online_token = data.get("online_access_info", {}).get("access_token")

        if not offline_token:
            return JsonResponse({"error": "OAuth failed", "details": data}, status=400)

        # --- NEW LOGIC START FROM LIVE: Fetch Shop Details ---
        shop_details_url = f"https://{shop}/admin/api/2024-01/shop.json"
        headers = {"X-Shopify-Access-Token": offline_token}
        detail_response = requests.get(shop_details_url, headers=headers)
        
        primary_custom_domain = None
        actual_shop_name = None
        shop_email = None

        if detail_response.status_code == 200:
            shop_data = detail_response.json().get('shop', {})
            # Extract data
            primary_custom_domain = shop_data.get('domain') 
            actual_shop_name = shop_data.get('name')
            shop_email = shop_data.get('email') 
        # --- NEW LOGIC END ---

        # Save/reactivate shop with NEW fields
        shop_obj, created = Shop.objects.update_or_create(
            domain=shop,
            defaults={
                "offline_token": offline_token,
                "online_token": online_token,
                "custom_domain": primary_custom_domain, # Saved custom domain
                "shop_name": actual_shop_name,          # Saved name
                "email": shop_email,                    # Saved email
                "is_active": True,
            },
        )

        # Register uninstall webhook
        register_uninstall_webhook(shop, offline_token)

        # Register GDPR webhooks (Added from Live)
        register_gdpr_webhooks(shop, offline_token)

        # --- Register orders/paid webhook for notification system (KEPT LOCAL LOGIC) ---
        register_orders_updated_webhook(shop, offline_token)

        # --- NEW: Register Shop Update Webhook (Syncs domain changes) ---
        register_shop_update_webhook(shop, offline_token)

        # Render install page (metafield creation now manual)
        return render(
            request,
            "recommender/shopify_install_page.html",
            {"shop": shop, "theme_editor_link": shop_obj.theme_editor_link},
        )

    except Exception as e:
        print(f"[ERROR] Exception in oauth_callback: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"Server error: {e}"}, status=500)


def create_metafield(request):
    """
    Manually creates or retrieves the 'usage_duration' metafield definition
    when the merchant clicks the 'Create Metafield' button.
    """
    shop = request.GET.get("shop")
    if not shop:
        return render(request, "error.html", {"message": "Missing shop parameter"})

    try:
        shop_obj = Shop.objects.get(domain=shop, is_active=True)

        if not shop_obj.offline_token:
            messages.error(request, "⚠️ Missing offline token. Please reinstall the app.")
            return redirect(f"/app_entry/?shop={shop}")

        definition_id = create_or_get_metafield_definition(shop, shop_obj.offline_token, shop_obj)

        if definition_id:
            messages.success(request, "✅ Metafield created or already exists.")
            return redirect(f"/app_entry/?shop={shop}&metafield_created=1")
        else:
            messages.error(request, "⚠️ Failed to create metafield.")
            return redirect(f"/app_entry/?shop={shop}")

    except Shop.DoesNotExist:
        return redirect(f"/start_auth/?shop={shop}")
    except Exception as e:
        print(f"[ERROR] Exception in create_metafield: {e}")
        messages.error(request, f"⚠️ Error: {e}")
        return redirect(f"/app_entry/?shop={shop}")


def create_shopify_page(request):
    """
    Creates the Face Analyzer page and navigation link manually
    when merchant clicks the button.
    """
    shop = request.GET.get("shop")
    if not shop:
        return render(request, "error.html", {"message": "Missing shop parameter"})

    try:
        shop_obj = Shop.objects.get(domain=shop, is_active=True)
        page_content = PageContent.objects.first()
        if not page_content:
            page_content = PageContent(title="Face Analyzer", body="<h1>Face Analyzer</h1>")

        page, deep_link = create_page(
            shop,
            shop_obj.offline_token,
            title=page_content.title,
            body=page_content.body,
            api_key=SHOPIFY_API_KEY,
            block_type="Beautyxia", # UPDATED FROM LIVE (was "test")
        )

        if page:
            shop_obj.theme_editor_link = deep_link
            shop_obj.save()
            messages.success(request, "✅ Page created and added to menu successfully.")
        else:
            messages.error(request, "⚠️ Failed to create page or add to menu.")

        return redirect(f"/app_entry/?shop={shop}&page_created=1")

    except Shop.DoesNotExist:
        return redirect(f"/start_auth/?shop={shop}")


def start_auth(request):
    """
    Starts the Shopify OAuth installation flow.
    Redirects merchant to Shopify to approve the app.
    """
    try:
        shop = request.GET.get("shop")
        if not shop:
            return render(request, "error.html", {"message": "Missing shop parameter"})

        redirect_uri = settings.BASE_URL + "/auth/callback/"
        scopes = (
            "read_products,write_products,read_metafields,write_metafields,write_content,"
            "write_online_store_pages,read_online_store_pages,read_online_store_navigation,"
            "write_online_store_navigation,read_themes,write_themes,read_orders,write_orders"
        )

        auth_url = (
            f"https://{shop}/admin/oauth/authorize?"
            f"client_id={SHOPIFY_API_KEY}&"
            f"scope={scopes}&"
            f"redirect_uri={urllib.parse.quote(redirect_uri)}&"
            f"state=12345"
        )

        return redirect(auth_url)

    except Exception as e:
        print(f"[ERROR] Exception in start_auth: {e}")
        return render(request, "error.html", {"message": f"Server error: {e}"})


### docs & policies

def documentation(request):
    """
    Render the documentation.
    """
    return render(request, "recommender/documentation.html")


def privacy_policy(request):
    """
    Render the privacy_policy.
    """
    return render(request, "recommender/privacy-policy.html")



############# dashboard #############
from django.db.models import Count
from datetime import datetime, timedelta
from django.utils import timezone

@login_required
def dashboard(request):
    if not (request.user.is_staff or request.user.is_superuser):
        return redirect('staff_login')
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)

    # Get filter from GET params (domain filter)
    domain_filter = request.GET.get('domain', '')

    # Filter FaceAnalysis by domain if search is used
    analysis_qs = FaceAnalysis.objects.all()
    if domain_filter:
        analysis_qs = analysis_qs.filter(domain__icontains=domain_filter)

    # ---- Stats (KEPT LOCAL WITH PURCHASES) ----
    stats = {
        "main_visitors": Visitor.objects.count(),  # Main page visitors only
        "analysis_today": FaceAnalysis.objects.filter(timestamp__date=today).count(),
        "analysis_week": FaceAnalysis.objects.filter(timestamp__date__gte=week_ago).count(),
        "analysis_month": FaceAnalysis.objects.filter(timestamp__date__gte=month_ago).count(),
        "likes": Feedback.objects.filter(feedback_type="like").count(),
        "dislikes": Feedback.objects.filter(feedback_type="dislike").count(),
        "total_purchases": Purchase.objects.count(),
        "purchases_notified": Purchase.objects.filter(notified=True).count(),
        "purchases_not_notified": Purchase.objects.filter(notified=False).count(),
    }

    # Visitors trend (last 7 days)
    visitors_by_day = (
        Visitor.objects.filter(date__gte=week_ago)
        .values("date")
        .annotate(count=Count("id"))
        .order_by("date")
    )
    dates = [v["date"].strftime("%Y-%m-%d") for v in visitors_by_day]
    counts = [v["count"] for v in visitors_by_day]

    analysis_today_hours = []
    analysis_today_counts = []
    for hour in range(24):
        hour_start = timezone.make_aware(datetime.combine(today, datetime.min.time()) + timedelta(hours=hour))
        hour_end = hour_start + timedelta(hours=1)
        count = FaceAnalysis.objects.filter(timestamp__gte=hour_start, timestamp__lt=hour_end).count()
        if count > 0 or hour <= timezone.now().hour:  # Only show up to current hour
            analysis_today_hours.append(f"{hour:02d}:00")
            analysis_today_counts.append(count)
    
    analysis_week_dates = []
    analysis_week_counts = []
    for i in range(7):
        date = today - timedelta(days=6-i)
        count = FaceAnalysis.objects.filter(timestamp__date=date).count()
        analysis_week_dates.append(date.strftime("%m/%d"))
        analysis_week_counts.append(count)
    
    analysis_month_dates = []
    analysis_month_counts = []
    for i in range(10):  # 10 groups of 3 days
        end_date = today - timedelta(days=i*3)
        start_date = end_date - timedelta(days=2)
        count = FaceAnalysis.objects.filter(
            timestamp__date__gte=start_date,
            timestamp__date__lte=end_date
        ).count()
        analysis_month_dates.insert(0, f"{start_date.strftime('%m/%d')}-{end_date.strftime('%m/%d')}")
        analysis_month_counts.insert(0, count)

    # Feedback ratio
    feedback_data = [
        Feedback.objects.filter(feedback_type="like").count(),
        Feedback.objects.filter(feedback_type="dislike").count(),
    ]

    # Top 5 domains
    top_domains = (
        FaceAnalysis.objects.values("domain")
        .annotate(total=Count("id"))
        .order_by("-total")[:5]
    )

    # All domains (for filter results)
    domain_stats = (
        analysis_qs.values("domain")
        .annotate(total=Count("id"))
        .order_by("-total")
    )

    context = {
        "stats": stats,
        "dates": dates,
        "counts": counts,
        "feedback_data": feedback_data,
        "top_domains": top_domains,
        "domain_stats": domain_stats,
        "domain_filter": domain_filter,
        "analysis_today_labels": analysis_today_hours,
        "analysis_today_data": analysis_today_counts,
        "analysis_week_labels": analysis_week_dates,
        "analysis_week_data": analysis_week_counts,
        "analysis_month_labels": analysis_month_dates,
        "analysis_month_data": analysis_month_counts,
    }
    return render(request, "recommender/dashboard.html", context)

#to search face analysis by domain
@login_required
def search_domains(request):
    domain_filter = request.GET.get('domain', '')
    analysis_qs = FaceAnalysis.objects.all()

    if domain_filter:
        analysis_qs = analysis_qs.filter(domain__icontains=domain_filter)

    domain_stats = (
        analysis_qs.values("domain")
        .annotate(total=Count("id"))
        .order_by("-total")
    )

    return JsonResponse({"domains": list(domain_stats)})


def staff_login(request):
    # Already logged in? Send to dashboard
    if request.user.is_authenticated:
        return redirect('dashboard')

    error = None

    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            if user.is_staff or user.is_superuser:
                login(request, user)
                return redirect('dashboard')
            else:
                error = "You are not authorized to access the staff dashboard."
        else:
            error = "Invalid username or password."

    return render(request, 'recommender/login.html', {'error': error})


def staff_logout(request):
    logout(request)
    return redirect('staff_login')