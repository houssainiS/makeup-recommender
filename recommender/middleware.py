from .models import Visitor, AllowedOrigin, Shop
from wordPress.models import WordpressShop # <--- Import WordpressShop
from django.utils.deprecation import MiddlewareMixin
from django.utils.timezone import now
from django.conf import settings
from django.core.cache import cache

class VisitorTrackingMiddleware(MiddlewareMixin):
    def process_request(self, request):
        # Skip middleware for webhooks and OAuth callback
        if request.path.startswith("/webhooks/") or request.path.startswith("/auth/callback"):
            return  # Do nothing for these paths

        # Cache CORS_ALLOWED_ORIGINS for 5 min instead of every request
        cors_urls = cache.get("allowed_origins")
        if cors_urls is None:
            try:
                # 1. Manual Origins (e.g., localhost, your own frontend)
                allowed_from_model = list(AllowedOrigin.objects.values_list("url", flat=True))
                
                # 2. Shopify Shops (Stored as 'domain.myshopify.com', so we add https://)
                allowed_from_shops = [
                    f"https://{domain}"
                    for domain in Shop.objects.filter(is_active=True).values_list("domain", flat=True)
                ]

                # 3. WordPress Shops (Stored as full URL 'https://site.com', just strip trailing slash)
                allowed_from_wp = [
                    url.rstrip('/') 
                    for url in WordpressShop.objects.filter(is_active=True).values_list("domain", flat=True)
                ]

                # Merge all lists + deduplicate
                cors_urls = list(set(allowed_from_model + allowed_from_shops + allowed_from_wp))
                
                cache.set("allowed_origins", cors_urls, 300)  # cache 5 min
            except Exception:
                cors_urls = []

        settings.CORS_ALLOWED_ORIGINS = cors_urls

        # Only create session if needed (for visitor tracking)
        if not request.session.session_key and request.path.startswith("/analyze"):
            request.session.create()

        # Log unique visitors per day
        if request.session.session_key:
            ip = self.get_client_ip(request)
            device_type = self.get_device_type(request)
            today = now().date()
            if not Visitor.objects.filter(session_key=request.session.session_key, date=today).exists():
                Visitor.objects.create(
                    session_key=request.session.session_key,
                    ip_address=ip,
                    device_type=device_type,
                    date=today
                )

    def process_response(self, request, response):
        """
        Add headers to allow Shopify embedded app in iframe.
        """
        if not request.path.startswith("/webhooks/"):
            # Allow embedding in Shopify admin
            response["X-Frame-Options"] = "ALLOWALL"
            
            # Note: If you ever want to embed this in WordPress Admin (iframe), 
            # you will need to add the WP domains here too. 
            # For now, this is safe as long as WP uses the redirect method.
            response["Content-Security-Policy"] = (
                "frame-ancestors https://*.myshopify.com https://admin.shopify.com;"
            )
        return response

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0]
        return request.META.get("REMOTE_ADDR")

    def get_device_type(self, request):
        user_agent = request.META.get("HTTP_USER_AGENT", "").lower()
        if "mobile" in user_agent:
            return "Mobile"
        elif "tablet" in user_agent:
            return "Tablet"
        return "Desktop"