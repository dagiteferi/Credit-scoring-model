from rest_framework import serializers

class ScoringSerializer(serializers.Serializer):
    feature1 = serializers.FloatField()
    feature2 = serializers.FloatField()
    # Add other features as needed
