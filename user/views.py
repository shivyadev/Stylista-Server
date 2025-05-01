from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.response import Response
from ai_models.models import UserInput
from ai_models.serializer import UserInputSerializer

@api_view(["POST"])
def get_user_outfits(request):
    try:
        if id in request.data:
            id = request.data.get("id")
        objects = UserInput.objects.filter(user_id=id)
        
        if not objects.exists():
            return Response({"message": "No outfits found for this user"}, status=status.HTTP_404_NOT_FOUND)

        serializer = UserInputSerializer(objects, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"message" : "Some error occured"}, e)