from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
from .serializers import ScoringSerializer

class ScoringView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = joblib.load('path_to_your_model.pkl')
    
    def post(self, request):
        serializer = ScoringSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            input_data = [data['feature1'], data['feature2']]  # Add other features as needed
            prediction = self.model.predict([input_data])
            return Response({'prediction': prediction[0]}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
