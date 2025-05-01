from django.urls import path
from .views import test, test_models, model_test, provide_outfits

urlpatterns = [
    path("test/", test, name="test"),
    path("test_model/", test_models, name="test_model"),
    path("model_test/", model_test, name="model_test"),
    path("provide_outfits/", provide_outfits, name="provide_outfits")
]
