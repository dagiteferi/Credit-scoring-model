# credit_scoring/views.py
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
from scoring_api.serializers import ScoringSerializer  # Update the import

class ScoringView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = joblib.load('C:/Users/HP/Documents/Dagii/Credit-scoring-model/models/Random Forest_best_model.pkl')  # Update path as needed
    
    def post(self, request):
        serializer = ScoringSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            input_data = [data['feature1'], data['feature2']]  # Add other features as needed
            prediction = self.model.predict([input_data])
            return Response({'prediction': prediction[0]}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# Define home_view separately, outside of any class
def home_view(request):
    return HttpResponse("Welcome to the Credit Scoring API!")
