from rest_framework import serializers
from api.models import ImageVideoModel, Coordinate, Project


class CoordinateSerializer(serializers.ModelSerializer):
    image_id = serializers.SerializerMethodField()

    class Meta:
        model = Coordinate
        fields = ["id", "x", "y", "image_id"]

    def get_image_id(self, obj):
        return obj.image.id


class ImageVideoModelSerializer(serializers.ModelSerializer):
    coordinates = CoordinateSerializer(many=True, read_only=True)

    class Meta:
        model = ImageVideoModel
        fields = [
            "id",
            "image",
            "uploaded_at",
            "coordinates",
            "is_label",
            "original_filename",
            "type",
            "duration",
            "frame_rate",
            "total_frames",
        ]


class ProjectSerializer(serializers.ModelSerializer):
    images = ImageVideoModelSerializer(many=True, read_only=True)

    class Meta:
        model = Project
        fields = ["id", "name", "type", "images"]
