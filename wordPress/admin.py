from django.contrib import admin
from .models import WordpressShop

@admin.register(WordpressShop)
class WordpressShopAdmin(admin.ModelAdmin):
    # Columns to show in the list view
    list_display = ('domain', 'plan_type', 'is_active', 'analysis_this_month', 'analysis_all_time', 'created_at')
    
    # Add filters on the right side
    list_filter = ('plan_type', 'is_active', 'created_at')
    
    # Search by domain or email
    search_fields = ('domain', 'admin_email', 'api_key')
    
    # Organize fields in the detail view
    fieldsets = (
        ('Site Info', {
            'fields': ('domain', 'admin_email', 'api_key')
        }),
        ('Subscription', {
            'fields': ('plan_type', 'is_active', 'monthly_limit')
        }),
        ('Usage Stats', {
            'fields': ('analysis_this_month', 'analysis_all_time')
        }),
        ('Technical Details', {
            'classes': ('collapse',), # Hide by default
            'fields': ('wp_version', 'plugin_version'),
        }),
    )
    
    # Make the API key read-only if you don't want to accidentally change it
    readonly_fields = ('created_at', 'last_heartbeat')