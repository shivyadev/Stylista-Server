from rest_framework import serializers
from .models import Test, UserInput
import requests

class TestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Test
        fields = "__all__"

    
class UserInputSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserInput
        fields = ['id', 'user_id', 'image_url', 'usage', 'gender', 'outfits']
        read_only_fields = ['image_url', 'outfits']

    def validate_outfits(self, value):
        if not isinstance(value, list) or not all(isinstance(row, list) and all(isinstance(num, int) for num in row) for row in value):
            raise serializers.ValidationError("Outfits must be a 2D array of integers.")
        return value