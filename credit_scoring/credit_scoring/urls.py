# credit_scoring/urls.py
from django.contrib import admin
from django.urls import path, include
from .views import home_view  # Correct import for home_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('scoring_api.urls')),
    path('', home_view, name='home'),  # Use home_view for the root URL
]
