from django.db import models

# Create your models here.

class Test(models.Model):
    name = models.CharField(max_length=100)

class UserInput(models.Model):
    user_id = models.CharField(max_length=100)
    image_url = models.CharField(max_length=255)  # Image URL will be set by the server
    usage = models.CharField(max_length=100)
    gender = models.CharField(max_length=20)
    outfits = models.JSONField(default=list)  # Stores a 2D array of integers
