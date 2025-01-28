# credit_scoring/views.py
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
from scoring_api.serializers import ScoringSerializer

class ScoringView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = joblib.load('C:/Users/HP/Documents/Dagii/Credit-scoring-model/models/Random Forest_best_model.pkl')  # Update path as needed
 
    def post(self, request):
        serializer = ScoringSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            input_data = [
                data['Std_Transaction_Amount'],
                data['CurrencyCode'],
                data['CountryCode'],
                data['ProviderId'],
                data['ProductId'],
                data['TransactionHour'],
                data['TransactionDay'],
                data['TransactionMonth'],
                data['TransactionWeekday'],
                data['CurrencyCode_WOE'],
                data['ProviderId_WOE'],
                data['ProductId_WOE'],
                data['ProductCategory_WOE'],
                data['Recency'],
                data['RFMS_score']
            ]
            try:
                prediction = self.model.predict([input_data])
                return Response({'prediction': prediction[0]}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            print(serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def home_view(request):
    return HttpResponse("Welcome to the Credit Scoring API!")
