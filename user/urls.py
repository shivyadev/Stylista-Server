from django.urls import path
from .views import get_user_outfits

urlpatterns = [
    path("all_outfits/", get_user_outfits, name="all_outfits"),
]
