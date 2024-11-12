import os
import cv2
import numpy as np
import torch
from django.conf import settings
from django.db import transaction
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from segment_anything import SamPredictor, sam_model_registry
from api.models import Coordinate, ImageModel
from api.serializers import ImageModelSerializer, CoordinateSerializer

sam = None
predictor = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageModel.objects.all()
    serializer_class = ImageModelSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    filter_backends = [filters.SearchFilter]
    search_fields = ["folder_path"]

    def create(self, request, *args, **kwargs):
        folder_path = request.data.get("folder_path", "default_folder")
        images = request.FILES.getlist("images")
        image_data = []

        for image in images:
            relative_path = os.path.join(folder_path, image.name)
            full_path = os.path.join(settings.MEDIA_ROOT, relative_path)

            image_exists = ImageModel.objects.filter(image=relative_path).first()

            if image_exists:
                image_record = image_exists
            else:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "wb+") as destination:
                    for chunk in image.chunks():
                        destination.write(chunk)

                image_record = ImageModel.objects.create(
                    image=relative_path,
                    folder_path=folder_path,
                )
            coordinates_qs = image_record.coordinates.all()
            coordinates = [{"x": coord.x, "y": coord.y} for coord in coordinates_qs]
            coordinates = coordinates[0] if coordinates else []
            image_data.append(
                {
                    "id": image_record.id,
                    "url": request.build_absolute_uri(
                        settings.MEDIA_URL + relative_path
                    ),
                    "filename": image.name,
                    "coordinates": coordinates,
                }
            )

        return Response({"images": image_data})

    @action(detail=False, methods=["get"])
    def folder_coordinates(self, request):
        folder_path = request.query_params.get("folder_path")
        if not folder_path:
            return Response({"error": "folder_path query parameter is required"}, status=400)
        
        images = ImageModel.objects.filter(folder_path=folder_path)
        all_coordinates = []
        for image in images:
            coordinates = image.coordinates.all()
            serializer = CoordinateSerializer(coordinates, many=True)
            all_coordinates.extend(serializer.data)

        return Response(all_coordinates)

    @action(detail=False, methods=["post"])
    def save_all_coordinates(self, request):
        all_coordinates = request.data.get("all_coordinates", [])
        if not all_coordinates:
            return Response(
                {"error": "No coordinates provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        with transaction.atomic():
            for item in all_coordinates:
                image_id = item.get("image_id")
                coordinates = item.get("coordinates", [])

                # Validate coordinates exist
                if not coordinates:
                    continue

                # Fetch the image
                try:
                    image = ImageModel.objects.get(id=image_id)
                except ImageModel.DoesNotExist:
                    continue

                # Delete existing coordinates for the image
                image.coordinates.all().delete()

                # Create new coordinate objects
                coordinate_objs = [
                    Coordinate(image=image, x=coord["x"], y=coord["y"])
                    for coord in coordinates
                ]
                Coordinate.objects.bulk_create(coordinate_objs)

        return Response({"status": "success"}, status=status.HTTP_200_OK)

    @action(detail=True, methods=["post"])
    def generate_mask(self, request, pk=None):
        image = self.get_object()
        global sam, predictor
        if sam is None:
            self.load_model()
        data = request.data
        coordinates = data.get("coordinates")
        prev_mask = data.get("mask_input")

        image_path = image.image.path
        img = cv2.imread(image_path)
        predictor.set_image(img)

        input_points = np.array([[coord["x"], coord["y"]] for coord in coordinates])
        input_labels = np.array(
            [1 if coord.get("include", True) else 0 for coord in coordinates]
        )

        mask_input = None
        if prev_mask:
            mask_input = np.array(prev_mask)

        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input=mask_input[None, :, :] if mask_input is not None else None,
            multimask_output=False,
        )

        binary_mask = masks[0].astype(np.uint8)
        return Response({"mask": binary_mask.tolist()})

    def load_model(self):
        global sam, predictor
        if sam is None:
            try:
                sam_checkpoint = "./ml_models/sam_models/sam_vit_h_4b8939.pth"
                model_type = "vit_h"
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=device)
                predictor = SamPredictor(sam)
            except Exception as e:
                raise e


class ModelManagerViewSet(viewsets.ViewSet):
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    @action(detail=False, methods=["post"])
    def load_model(self, request):
        global sam, predictor
        if sam is None:
            try:
                sam_checkpoint = "./ml_models/sam_models/sam_vit_h_4b8939.pth"
                model_type = "vit_h"
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=device)
                predictor = SamPredictor(sam)
                return Response({"message": "Model loaded successfully"})
            except Exception as e:
                return Response(
                    {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        else:
            return Response({"message": "Model is already loaded"})

    @action(detail=False, methods=["post"])
    def unload_model(self, request):
        global sam, predictor
        if sam is not None:
            sam = None
            predictor = None
            torch.cuda.empty_cache()
            return Response({"message": "Model unloaded successfully"})
        else:
            return Response({"message": "Model is not loaded"})
