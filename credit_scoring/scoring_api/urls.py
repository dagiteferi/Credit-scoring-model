from django.urls import path
from .views import ScoringView

urlpatterns = [
    path('predict/', ScoringView.as_view(), name='predict'),
]
