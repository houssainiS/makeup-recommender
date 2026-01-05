from django.urls import path
from . import views

urlpatterns = [
    path('connect/', views.connect_page, name='wp-connect'),
    path('finalize/', views.finalize_connection, name='wp-finalize'),
    path('deactivate/', views.deactivate_shop, name='wp-deactivate'), # New path
]