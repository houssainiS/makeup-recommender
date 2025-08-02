from django.contrib import admin
from .models import Visitor, FaceAnalysis

@admin.register(Visitor)
class VisitorAdmin(admin.ModelAdmin):
    list_display = ('ip_address', 'device_type', 'date')
    list_filter = ('device_type', 'date')
    search_fields = ('ip_address',)


@admin.register(FaceAnalysis)
class FaceAnalysisAdmin(admin.ModelAdmin):
    list_display = ('ip_address', 'device_type', 'timestamp')
    list_filter = ('device_type', 'timestamp')
    search_fields = ('ip_address',)
