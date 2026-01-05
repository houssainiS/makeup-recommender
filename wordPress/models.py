from django.db import models

class WordpressShop(models.Model):
    # Identity
    domain = models.URLField(unique=True)
    admin_email = models.EmailField()
    api_key = models.CharField(max_length=64, unique=True)
    
    # Plan & Status
    PLAN_CHOICES = [('free', 'Free'), ('pro', 'Pro'), ('enterprise', 'Enterprise')]
    plan_type = models.CharField(max_length=20, choices=PLAN_CHOICES, default='free')
    is_active = models.BooleanField(default=True)
    
    # Analytics & Quotas
    monthly_limit = models.IntegerField(default=100)
    analysis_this_month = models.IntegerField(default=0)
    analysis_all_time = models.BigIntegerField(default=0)
    
    # Metadata for Debugging
    wp_version = models.CharField(max_length=10, blank=True)
    plugin_version = models.CharField(max_length=10, blank=True)
    last_heartbeat = models.DateTimeField(auto_now=True) # Last time they used the plugin
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.domain