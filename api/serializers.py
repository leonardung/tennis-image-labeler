from rest_framework import serializers
from api.models import ImageModel, Coordinate


class CoordinateSerializer(serializers.ModelSerializer):
    image_id = serializers.SerializerMethodField()

    class Meta:
        model = Coordinate
        fields = ["id", "x", "y", "image_id"]

    def get_image_id(self, obj):
        return obj.image.id 


class ImageModelSerializer(serializers.ModelSerializer):
    coordinates = CoordinateSerializer(many=True, read_only=True)

    class Meta:
        model = ImageModel
        fields = ["id", "image", "folder_path", "uploaded_at", "coordinates"]
