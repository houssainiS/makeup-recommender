from django.contrib import admin
from .models import Visitor, FaceAnalysis , Feedback

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


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('feedback_type', 'dislike_reason', 'created_at')
    list_filter = ('feedback_type', 'created_at')
    search_fields = ('dislike_reason',)
    readonly_fields = ('created_at',)