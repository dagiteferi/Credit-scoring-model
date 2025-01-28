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
        try:
            serializer = ScoringSerializer(data=request.data)
            if serializer.is_valid():
                data = serializer.validated_data

                required_fields = [
                    'ProviderId', 'ProductCategory', 'Amount', 'Value', 'PricingStrategy', 
                    'FraudResult', 'Total_Transaction_Amount', 'Average_Transaction_Amount', 
                    'Transaction_Count', 'Std_Transaction_Amount', 'Transaction_Hour', 'Transaction_Day', 
                    'Transaction_Month', 'Transaction_Year', 'CurrencyCode_WOE', 'ProviderId_WOE', 
                    'ProductId_WOE', 'ProductCategory_WOE', 'Recency', 'RFMS_score', 'ProductId_1', 
                    'ProductId_2', 'ProductId_3', 'ProductId_4', 'ProductId_5', 'ProductId_6', 
                    'ProductId_7', 'ProductId_8', 'ProductId_9', 'ProductId_10', 'ProductId_11', 
                    'ProductId_12', 'ProductId_13', 'ProductId_14', 'ProductId_15', 'ProductId_16', 
                    'ProductId_17', 'ProductId_18', 'ProductId_19', 'ProductId_20', 'ProductId_21', 
                    'ProductId_22', 'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5', 
                    'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionWeekday', 
                     'ProductId'
                ]

                # Check if any required fields are missing
                for field in required_fields:
                    if field not in data:
                        return Response({'error': f'Missing field: {field}'}, status=status.HTTP_400_BAD_REQUEST)

                input_data = [data[field] for field in required_fields]

                prediction = self.model.predict([input_data])
                return Response({'prediction': prediction[0]}, status=status.HTTP_200_OK)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def home_view(request):
    return HttpResponse("Welcome to the Credit Scoring API!")