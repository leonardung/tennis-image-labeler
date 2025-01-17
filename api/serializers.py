from rest_framework import serializers

from api.models.coordinate import Coordinate
from api.models.image import ImageModel
from api.models.project import Project


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
        fields = [
            "id",
            "image",
            "thumbnail",
            "mask",
            "uploaded_at",
            "coordinates",
            "is_label",
            "original_filename",
        ]


class ProjectSerializer(serializers.ModelSerializer):
    images = ImageModelSerializer(many=True, read_only=True)

    class Meta:
        model = Project
        fields = ["id", "name", "type", "images"]
