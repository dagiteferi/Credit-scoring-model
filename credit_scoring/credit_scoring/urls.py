from django.urls import path
from .views import ScoringView, home_view

urlpatterns = [
    path('', home_view, name='home'),
    path('api/predict/', ScoringView.as_view(), name='predict'),
]