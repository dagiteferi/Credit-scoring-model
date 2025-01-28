from django.urls import path
from .views import home_view, get_score

urlpatterns = [
    path('', home_view, name='home'),
    path('get_score/', get_score, name='get_score'),
]
