# scoring_api/urls.py
from django.urls import path
from .views import ScoringView  # Only import ScoringView

urlpatterns = [
    path('predict/', ScoringView.as_view(), name='predict'),
]
