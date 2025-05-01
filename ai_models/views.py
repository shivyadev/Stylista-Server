from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.response import Response
from .serializer import TestSerializer, UserInputSerializer
from django.conf import settings
from .utlis import predict_category, extract_cloth_colors_with_segmentation, extract_compatible_clothes, find_closest_color, upload_image
from PIL import Image
from io import BytesIO

@api_view(["POST"])
def test(request):
    serializer = TestSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response({"message": "Saved"}, status=status.HTTP_201_CREATED)
    return Response({"message": "Error"}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def test_models(request):
    """Test if all models and category mappings are loaded properly."""
    models_loaded = {
        "ResNet50": "Loaded" if "resnet50" in settings.MODEL_REGISTRY else "Not Loaded",
        "EfficientNet": "Loaded" if "efficientnet" in settings.MODEL_REGISTRY else "Not Loaded",
        "Mask R-CNN": "Loaded" if "mask_rcnn" in settings.MODEL_REGISTRY else "Not Loaded",
        "Casual Model": "Loaded" if "casual_model" in settings.MODEL_REGISTRY else "Not Loaded",
        "Formal Model": "Loaded" if "formal_model" in settings.MODEL_REGISTRY else "Not Loaded",
        "Sports Model": "Loaded" if "sports_model" in settings.MODEL_REGISTRY else "Not Loaded",
        "Casual Mapping": "Loaded" if "casual_mapping" in settings.MODEL_REGISTRY else "Not Loaded",
        "Formal Mapping": "Loaded" if "formal_mapping" in settings.MODEL_REGISTRY else "Not Loaded",
        "Sports Mapping": "Loaded" if "sports_mapping" in settings.MODEL_REGISTRY else "Not Loaded",
    }

    return Response({"models_status": models_loaded}) 

@api_view(["POST"])
def model_test(request):
    """
    Handles an image upload and predicts the clothing category.
    """
    try:
        # Ensure an image file is provided in the request
        if "image" not in request.FILES:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Read image from request
        image_file = request.FILES["image"]
        image = Image.open(image_file).convert("RGB")  # Convert to RGB

        # Load the model and category mapping from settings
        casual_model = settings.MODEL_REGISTRY["casual_model"]
        casual_mapping = settings.MODEL_REGISTRY["casual_mapping"]

        # Predict category
        category, _ = predict_category(image, casual_model, casual_mapping)

        return Response({"Category": category}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

@api_view(["POST"])
def provide_outfits(request):
    serializer = UserInputSerializer(data=request.data)

    try:
        if "image" not in request.FILES:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES["image"]
        image_bytes = image_file.read()
        image_file.seek(0)

         # Read image from request
        if "gender" not in request.data:
            return Response({"error": "No gender provided"}, status=status.HTTP_400_BAD_REQUEST)
        gender = request.data["gender"].replace(" ", "")

        if "usage" not in request.data:
            return Response({"error": "No usage provided"}, status=status.HTTP_400_BAD_REQUEST)
        usage = request.data["usage"].replace(" ", "")

        pairs_dataset = 'https://fashion-recommendation-models.s3.ap-south-1.amazonaws.com/Compatible-Outfits.csv'
        clothes_dataset = "https://fashion-recommendation-models.s3.ap-south-1.amazonaws.com/filtered_data_13.csv"

        casual_model = settings.MODEL_REGISTRY["casual_model"]
        casual_mapping = settings.MODEL_REGISTRY["casual_mapping"]
        formal_model = settings.MODEL_REGISTRY["formal_model"]
        formal_mapping = settings.MODEL_REGISTRY["formal_mapping"]
        sports_model = settings.MODEL_REGISTRY["sports_model"]
        sports_mapping = settings.MODEL_REGISTRY["sports_mapping"]
        

        category = ""
        if usage == "Casual":
            category = predict_category(image_bytes, casual_model, casual_mapping)[0]
        elif usage == "Formal":
            category = predict_category(image_bytes, formal_model, formal_mapping)[0]
        elif usage == "Sports":
            category = predict_category(image_bytes, sports_model, sports_mapping)[0]
        else: 
            print("Unrecognized Usage")
            return Response({"Error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        print(category)

        ArticleType = {
            "Topwear" : ['Tshirts','Shirts','Tops','Kurtas','Kurtis','Dresses'],
            "Layered Wear":['Jackets','Waistcoat','Sweatshirts','Blazers','Shrug'],
            "Bottomwear" :['Jeans','Trousers','Track Pants','Shorts','Capris','Leggings','Skirts'],
            "Footwear" : ['Casual Shoes','Formal Shoes','Sports Shoes','Sneakers','Flats','Loafers','Heels','Sandal'],
            "Accessories": ['Watches','Handbags','Socks', 'Belts']
        }

        cloth_type = ""

        for key, values in ArticleType.items():
            if category in values:
                cloth_type = key

        
        colors, segmented_image =  extract_cloth_colors_with_segmentation(image_bytes)
        color = tuple(colors[0][0])
        column_name = f"{cloth_type} Color RGB"
        
        closest_matches = find_closest_color(pairs_dataset, color, column_name)
        filtered_matches = closest_matches.loc[(closest_matches[cloth_type] == category) & (closest_matches["Usage"].str.contains(usage))].head(10) 
        
        top_matches = extract_compatible_clothes(clothes_dataset, filtered_matches, cloth_type, gender, usage)
        top_matches = [[int(value) for value in row] for row in top_matches]

        if serializer.is_valid():
            image_url = upload_image(image_file)

            # Create an instance manually since 'outfits' is read-only
            instance = serializer.save(image_url=image_url)  

            # Assign the generated outfits and save the instance
            instance.outfits = top_matches  
            instance.save()

            print("Saving data with outfits")
        
        return Response({"matches": top_matches}, status=status.HTTP_200_OK) 
        
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)