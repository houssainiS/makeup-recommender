import uuid
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.cache import cache # <--- Added for cache management
from .models import WordpressShop 

def connect_page(request):
    """
    Step 1: Show the user a 'Do you want to connect?' page.
    URL: /wordpress/connect/?shop_url=...&admin_email=...
    """
    shop_url = request.GET.get('shop_url')
    admin_email = request.GET.get('admin_email', '')

    if not shop_url:
        return render(request, 'error.html', {'message': 'Missing shop URL'})

    context = {
        'shop_url': shop_url,
        'admin_email': admin_email
    }
    return render(request, 'wordPress/confirm_connect.html', context)

def finalize_connection(request):
    """
    Step 2: User clicked 'Accept'. Create or Update record and redirect back.
    """
    if request.method == "POST":
        shop_url = request.POST.get('shop_url')
        admin_email = request.POST.get('admin_email')

        # Generate a fresh API Key every time they connect (Key Rotation)
        new_api_key = uuid.uuid4().hex + uuid.uuid4().hex 
        
        # update_or_create will:
        # 1. Find the shop by 'domain'
        # 2. If found -> UPDATE the api_key and email
        # 3. If not found -> CREATE a new record
        shop, created = WordpressShop.objects.update_or_create(
            domain=shop_url,
            defaults={
                'api_key': new_api_key,     
                'admin_email': admin_email,
                'is_active': True
            }
        )

        # IMPORTANT: Clear the middleware cache so this shop is allowed immediately
        cache.delete("allowed_origins") 

        # Redirect back to WP with the NEW key
        callback_url = f"{shop_url}/wp-admin/admin.php?page=face-analyzer&status=success&api_key={new_api_key}"
        return redirect(callback_url)
    
    return redirect('home')

@csrf_exempt  # Exempt because this is an API call from WordPress
def deactivate_shop(request):
    """
    Called by WordPress when the 'Disconnect' button is clicked.
    """
    if request.method == "POST":
        shop_url = request.POST.get('shop_url')
        api_key = request.POST.get('api_key')

        # Find the shop and mark as inactive
        shop = WordpressShop.objects.filter(domain=shop_url, api_key=api_key).first()
        if shop:
            shop.is_active = False
            shop.save()
            
            # IMPORTANT: Clear the middleware cache so access is revoked immediately
            cache.delete("allowed_origins")
            
            return JsonResponse({'status': 'success', 'message': 'Shop deactivated'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)