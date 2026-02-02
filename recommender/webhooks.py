# webhooks.py
import requests
import hmac
import hashlib
import base64
import json
import os
import traceback
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.conf import settings
from .models import Shop, Purchase

# ======================================================
# LOAD SHOPIFY API SECRET
# ======================================================
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET", "fallback-secret-for-dev")


# ======================================================
# HMAC VERIFICATION FUNCTION
# ======================================================
def verify_webhook(data, hmac_header):
    """
    Verifies Shopify webhook HMAC to ensure request authenticity.
    """
    digest = hmac.new(
        SHOPIFY_API_SECRET.encode("utf-8"),
        data,
        hashlib.sha256
    ).digest()
    calculated_hmac = base64.b64encode(digest).decode()
    return hmac.compare_digest(calculated_hmac, hmac_header)


# ======================================================
# APP UNINSTALLED WEBHOOK
# ======================================================
@csrf_exempt
def app_uninstalled(request):
    """
    Handles Shopify 'app/uninstalled' webhook.
    Marks the shop as inactive and deletes its saved metafield definition.
    Also removes any stored theme editor deep link.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    # --- Verify webhook authenticity ---
    hmac_header = request.headers.get("X-Shopify-Hmac-Sha256")
    if not hmac_header:
        print("[Webhook] Missing HMAC header")
        return JsonResponse({"error": "Missing HMAC header"}, status=400)

    if not verify_webhook(request.body, hmac_header):
        print("[Webhook] HMAC verification failed")
        return JsonResponse({"error": "Invalid webhook"}, status=401)

    # --- Get shop domain ---
    shop_domain = request.headers.get("X-Shopify-Shop-Domain")
    if not shop_domain:
        return JsonResponse({"error": "Missing shop domain"}, status=400)

    try:
        shop_obj = Shop.objects.filter(domain=shop_domain).first()
        if not shop_obj:
            print(f"[Webhook] No shop record found for {shop_domain}")
            return JsonResponse({"status": "ok"}, status=200)

        # Shopify GraphQL endpoint
        graphql_url = f"https://{shop_domain}/admin/api/2025-07/graphql.json"
        headers = {
            "X-Shopify-Access-Token": shop_obj.offline_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # --- Step 1: Get metafield definition ID if not saved ---
        def_id = shop_obj.metafield_definition_id
        if not def_id:
            query = """
            {
              metafieldDefinitions(first: 10, ownerType: PRODUCT, namespace: "custom", key: "usage_duration") {
                edges { node { id } }
              }
            }
            """
            resp = requests.post(graphql_url, headers=headers, json={"query": query})
            edges = resp.json().get("data", {}).get("metafieldDefinitions", {}).get("edges", [])
            if edges:
                def_id = edges[0]["node"]["id"]
                print(f"[Webhook] Found definition ID via query: {def_id}")

        # --- Step 2: Delete metafield definition if found ---
        if def_id:
            delete_query = """
            mutation metafieldDefinitionDelete($id: ID!) {
              metafieldDefinitionDelete(id: $id) {
                deletedDefinitionId
                userErrors { field message }
              }
            }
            """
            variables = {"id": def_id}
            try:
                del_resp = requests.post(graphql_url, headers=headers, json={"query": delete_query, "variables": variables})
                print("[Webhook] Metafield definition delete response:", del_resp.json())
            except Exception as del_e:
                print("[Webhook] Failed to delete metafield definition:", del_e)
        else:
            print("[Webhook] No metafield definition to delete.")

        # --- Step 3: Mark shop inactive & clear theme editor link ---
        shop_obj.is_active = False
        shop_obj.theme_editor_link = None
        shop_obj.save(update_fields=["is_active", "theme_editor_link"])
        print(f"[Webhook] App uninstalled from {shop_domain}, marked inactive and cleared theme_editor_link")

    except Exception as e:
        print(f"[Webhook] Exception handling uninstall for {shop_domain}: {e}")

    return JsonResponse({"status": "ok"}, status=200)


# ======================================================
# REGISTER UNINSTALL WEBHOOK
# ======================================================
def register_uninstall_webhook(shop, access_token):
    url = f"https://{shop}/admin/api/2025-07/webhooks.json"
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json"
    }
    data = {
        "webhook": {
            "topic": "app/uninstalled",
            "address": "https://beautyai.duckdns.org/webhooks/app_uninstalled/",
            "format": "json"
        }
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        print("[Webhook Registration] Response:", response.json())
    except Exception as e:
        print("[Webhook Registration] Failed to register webhook:", str(e))


# ======================================================
# GDPR MANDATORY WEBHOOKS (Required for App Approval)
# ======================================================

@csrf_exempt
def customers_data_request(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    hmac_header = request.headers.get("X-Shopify-Hmac-Sha256")
    if not verify_webhook(request.body, hmac_header):
        print("[GDPR] Invalid HMAC for customers/data_request")
        return JsonResponse({"error": "Invalid webhook"}, status=401)

    data = json.loads(request.body.decode("utf-8"))
    print("[GDPR] Customer data request received:", data)
    return JsonResponse({"status": "ok"}, status=200)


@csrf_exempt
def customers_redact(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    hmac_header = request.headers.get("X-Shopify-Hmac-Sha256")
    if not verify_webhook(request.body, hmac_header):
        print("[GDPR] Invalid HMAC for customers/redact")
        return JsonResponse({"error": "Invalid webhook"}, status=401)

    data = json.loads(request.body.decode("utf-8"))
    print("[GDPR] Customer redact request received:", data)
    return JsonResponse({"status": "ok"}, status=200)


@csrf_exempt
def shop_redact(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    hmac_header = request.headers.get("X-Shopify-Hmac-Sha256")
    if not verify_webhook(request.body, hmac_header):
        print("[GDPR] Invalid HMAC for shop/redact")
        return JsonResponse({"error": "Invalid webhook"}, status=401)

    data = json.loads(request.body.decode("utf-8"))
    shop_domain = data.get("shop_domain")
    print(f"[GDPR] Shop redact request received for: {shop_domain}")
    return JsonResponse({"status": "ok"}, status=200)


def register_gdpr_webhooks(shop, access_token):
    topics = {
        "customers/data_request": "https://beautyai.duckdns.org/webhooks/customers_data_request/",
        "customers/redact": "https://beautyai.duckdns.org/webhooks/customers_redact/",
        "shop/redact": "https://beautyai.duckdns.org/webhooks/shop_redact/",
    }

    for topic, address in topics.items():
        url = f"https://{shop}/admin/api/2025-07/webhooks.json"
        headers = {
            "X-Shopify-Access-Token": access_token,
            "Content-Type": "application/json"
        }
        data = {
            "webhook": {
                "topic": topic,
                "address": address,
                "format": "json"
            }
        }
        try:
            response = requests.post(url, json=data, headers=headers)
            print(f"[GDPR Webhook Registration] {topic}: {response.json()}")
        except Exception as e:
            print(f"[GDPR Webhook Registration] Failed for {topic}:", str(e))


# ==========================================================
# 📌 LOCAL SPECIFIC: Orders/Updated Webhook & Logic
# ==========================================================

def fetch_usage_duration(product_id, shop_domain):
    """
    Fetch usage_duration metafield for a product via GraphQL.
    """
    try:
        shop = Shop.objects.get(domain=shop_domain)
        graphql_url = f"https://{shop_domain}/admin/api/2025-07/graphql.json"
        headers = {
            "X-Shopify-Access-Token": shop.offline_token,
            "Content-Type": "application/json",
        }
        query = """
        query($id: ID!) {
          product(id: $id) {
            metafield(namespace: "custom", key: "usage_duration") {
              value
            }
          }
        }
        """
        variables = {"id": f"gid://shopify/Product/{product_id}"}
        resp = requests.post(
            graphql_url,
            headers=headers,
            json={"query": query, "variables": variables}
        )
        value = resp.json().get("data", {}).get("product", {}).get("metafield", {}).get("value")
        return int(value) if value else 0
    except Exception as e:
        print(f"[Webhook] Failed to fetch usage_duration: {e}")
        return 0


def register_orders_updated_webhook(shop_domain, access_token):
    """
    Registers the 'orders/updated' webhook for a shop with detailed debugging.
    """
    url = f"https://{shop_domain}/admin/api/2025-07/webhooks.json"
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json"
    }
    data = {
        "webhook": {
            "topic": "orders/updated",
            "address": f"{settings.BASE_URL}/webhooks/order_updated/",
            "format": "json"
        }
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        print("[Webhook Registration] HTTP Status Code:", response.status_code)
        try:
            resp_json = response.json()
            print("[Webhook Registration] Response JSON:", resp_json)
        except ValueError:
            print("[Webhook Registration] Raw response text:", response.text)
    except requests.RequestException as req_err:
        print("[Webhook Registration] Request failed:", req_err)


@csrf_exempt
def order_updated(request):
    """
    Endpoint to receive Shopify 'orders/updated' webhook.
    Only saves purchase when financial_status == 'paid'.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    # --- DEBUG: Log Headers & Body ---
    print("[Webhook Debug] Headers:", dict(request.headers))
    try:
        body_text = request.body.decode()
        print("[Webhook Debug] Raw body:", body_text)
    except Exception as decode_err:
        print("[Webhook Debug] Failed to decode body:", decode_err)
        body_text = ""

    # --- Verify HMAC ---
    hmac_header = request.headers.get("X-Shopify-Hmac-Sha256")
    if not hmac_header or not verify_webhook(request.body, hmac_header):
        print("[Webhook Debug] Invalid HMAC or missing header")
        return JsonResponse({"error": "Invalid webhook"}, status=401)

    try:
        # Parse JSON
        data = {}
        if body_text:
            data = json.loads(body_text)

        # Get shop
        shop_domain = request.headers.get("X-Shopify-Shop-Domain")
        shop = Shop.objects.filter(domain=shop_domain).first()
        if not shop:
            print(f"[Webhook Debug] Shop not found: {shop_domain}")
            return JsonResponse({"error": "Shop not found"}, status=404)

        # Check financial_status
        financial_status = data.get("financial_status")
        if financial_status != "paid":
            print(f"[Webhook Debug] Order ignored, financial_status={financial_status}")
            return JsonResponse({"status": "ignored"}, status=200)

        # Extract order info
        email = data.get("email")
        phone = data.get("phone")
        order_id = data.get("id")
        line_items = data.get("line_items", [])

        print(f"[Webhook] Paid order {order_id} from {shop_domain}")

        for item in line_items:
            product_id = item.get("product_id")
            product_name = item.get("title")
            usage_days = fetch_usage_duration(product_id, shop_domain)

            if usage_days and usage_days > 0:
                Purchase.objects.create(
                    email=email,
                    phone=phone,
                    order_id=str(order_id),
                    product_id=str(product_id),
                    product_name=product_name,
                    purchase_date=timezone.now(),
                    usage_duration_days=usage_days,
                    domain=shop_domain,
                )
                print(f"[Webhook] ✅ Saved purchase: {product_name} ({usage_days} days)")
            else:
                print(f"[Webhook] ⚠️ Skipped {product_name} — usage_duration=0")

    except Exception as e:
        print("[Orders/Updated Webhook] Exception:", e)
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"status": "ok"}, status=200)

# ======================================================
# SHOP UPDATE WEBHOOK (Handle Domain/Email Changes)
# ======================================================
@csrf_exempt
def shop_updated(request):
    """
    Handles Shopify 'shop/update' webhook.
    Updates the local Shop model when the merchant changes their
    Primary Domain, Shop Name, or Email in Shopify settings.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    # 1. Verify HMAC
    hmac_header = request.headers.get("X-Shopify-Hmac-Sha256")
    if not hmac_header or not verify_webhook(request.body, hmac_header):
        print("[Shop Update] HMAC verification failed")
        return JsonResponse({"error": "Invalid webhook"}, status=401)

    # 2. Parse Data
    try:
        data = json.loads(request.body.decode("utf-8"))
        
        # The 'myshopify_domain' is our unique ID in the database
        myshopify_domain = data.get("myshopify_domain") 
        
        # The 'domain' field in the webhook is the "Primary Domain" (e.g., www.brand.com)
        new_primary_domain = data.get("domain")
        new_shop_name = data.get("name")
        new_email = data.get("email")
        
        print(f"[Shop Update] Received update for {myshopify_domain}")

        # 3. Update Database
        shop_obj = Shop.objects.filter(domain=myshopify_domain).first()
        
        if shop_obj:
            updated = False
            
            # Check for changes and update fields if necessary
            if shop_obj.custom_domain != new_primary_domain:
                shop_obj.custom_domain = new_primary_domain
                updated = True
                print(f" -> Updating custom_domain to: {new_primary_domain}")

            if shop_obj.shop_name != new_shop_name:
                shop_obj.shop_name = new_shop_name
                updated = True
                print(f" -> Updating shop_name to: {new_shop_name}")

            if shop_obj.email != new_email:
                shop_obj.email = new_email
                updated = True
                print(f" -> Updating email to: {new_email}")

            if updated:
                shop_obj.save()
                print("[Shop Update] Database updated successfully.")
            else:
                print("[Shop Update] No changes detected.")
        else:
            print(f"[Shop Update] Warning: Shop {myshopify_domain} not found in DB.")

        return JsonResponse({"status": "ok"}, status=200)

    except Exception as e:
        print(f"[Shop Update] Error processing webhook: {e}")
        return JsonResponse({"error": "Server error"}, status=500)


def register_shop_update_webhook(shop_domain, access_token):
    """
    Registers the 'shop/update' webhook.
    """
    url = f"https://{shop_domain}/admin/api/2025-07/webhooks.json"
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json"
    }
    # Update this address to match your live URL + the path defined in urls.py
    webhook_address = f"{settings.BASE_URL}/webhooks/shop_updated/"
    
    data = {
        "webhook": {
            "topic": "shop/update",
            "address": webhook_address,
            "format": "json"
        }
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 201:
             print(f"[Webhook Registration] 'shop/update' registered successfully for {shop_domain}")
        else:
             print(f"[Webhook Registration] 'shop/update' registration failed: {response.text}")
    except Exception as e:
        print("[Webhook Registration] Exception registering shop/update:", str(e))