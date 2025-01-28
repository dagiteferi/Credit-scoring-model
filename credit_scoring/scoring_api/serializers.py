# scoring_api/serializers.py
from rest_framework import serializers

class ScoringSerializer(serializers.Serializer):
    Std_Transaction_Amount = serializers.FloatField()
    CurrencyCode = serializers.IntegerField()
    CountryCode = serializers.IntegerField()
    ProviderId = serializers.IntegerField()
    ProductId = serializers.IntegerField()
    TransactionHour = serializers.IntegerField()
    TransactionDay = serializers.IntegerField()
    TransactionMonth = serializers.IntegerField()
    TransactionWeekday = serializers.IntegerField()
    CurrencyCode_WOE = serializers.FloatField()
    ProviderId_WOE = serializers.FloatField()
    ProductId_WOE = serializers.FloatField()
    ProductCategory_WOE = serializers.FloatField()
    Recency = serializers.IntegerField()
    RFMS_score = serializers.FloatField()
