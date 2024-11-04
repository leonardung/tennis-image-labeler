from rest_framework import serializers
from .models import ImageModel, Coordinate


class CoordinateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Coordinate
        fields = ("x", "y")

    def create(self, validated_data):
        image = self.context.get("image")
        if not image:
            raise serializers.ValidationError("Image context is required.")

        # Handle the case when multiple coordinates are provided
        if isinstance(validated_data, list):
            coordinates = [Coordinate(image=image, **item) for item in validated_data]
            return Coordinate.objects.bulk_create(coordinates)
        else:
            # For single coordinate
            return Coordinate.objects.create(image=image, **validated_data)


class ImageModelSerializer(serializers.ModelSerializer):
    coordinates = CoordinateSerializer(many=True, read_only=True)

    class Meta:
        model = ImageModel
        fields = ("id", "image", "folder_path", "uploaded_at", "coordinates")
