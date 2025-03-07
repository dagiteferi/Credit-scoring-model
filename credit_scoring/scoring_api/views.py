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
                data['ProviderId'], data['ProductCategory'], data['Amount'], data['Value'], data['PricingStrategy'], 
                data['FraudResult'], data['Total_Transaction_Amount'], data['Average_Transaction_Amount'], 
                data['Transaction_Count'], data['Std_Transaction_Amount'], data['Transaction_Hour'], data['Transaction_Day'], 
                data['Transaction_Month'], data['Transaction_Year'], data['CurrencyCode_WOE'], data['ProviderId_WOE'], 
                data['ProductId_WOE'], data['ProductCategory_WOE'], data['Recency'], data['RFMS_score'], data['ProductId_1'], 
                data['ProductId_2'], data['ProductId_3'], data['ProductId_4'], data['ProductId_5'], data['ProductId_6'], 
                data['ProductId_7'], data['ProductId_8'], data['ProductId_9'], data['ProductId_10'], data['ProductId_11'], 
                data['ProductId_12'], data['ProductId_13'], data['ProductId_14'], data['ProductId_15'], data['ProductId_16'], 
                data['ProductId_17'], data['ProductId_18'], data['ProductId_19'], data['ProductId_20'], data['ProductId_21'], 
                data['ProductId_22'], data['ChannelId_ChannelId_2'], data['ChannelId_ChannelId_3'], data['ChannelId_ChannelId_5'], 
                data['TransactionHour'], data['TransactionDay'], data['TransactionMonth'], data['TransactionWeekday'], 
                 data['ProductId']
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