# credit_scoring/urls.py
from django.urls import path
from .views import ScoringView, home_view

urlpatterns = [
    path('api/predict/', ScoringView.as_view(), name='predict'),
    path('', home_view, name='home'),
]
