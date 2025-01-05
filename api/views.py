import os
import cv2
import numpy as np
import torch
import ffmpeg
from django.conf import settings
from django.db import transaction
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from segment_anything import SamPredictor, sam_model_registry
from api.models import Coordinate, Project, ImageVideoModel
from api.serializers import (
    CoordinateSerializer,
    ProjectSerializer,
    ImageVideoModelSerializer,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

sam = None
predictor = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        return Project.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageVideoModel.objects.all()
    serializer_class = ImageVideoModelSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        return ImageVideoModel.objects.filter(project__user=self.request.user)

    def _get_video_metadata(self, video_file):
        """Extract video metadata using ffmpeg"""
        try:
            probe = ffmpeg.probe(video_file.temporary_file_path())
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            if video_stream:
                duration = float(video_stream['duration'])
                frame_rate = eval(video_stream['r_frame_rate'])  # e.g. '30/1' -> 30.0
                total_frames = int(float(video_stream['nb_frames']))
                return duration, frame_rate, total_frames
        except Exception as e:
            print(f"Error extracting video metadata: {e}")
        return None, None, None

    def create(self, request, *args, **kwargs):
        project_id = request.data.get("project_id")
        is_label = request.data.get("is_label", False)

        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return Response(
                {
                    "error": "Project not found or you do not have permission to access it."
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        images = request.FILES.getlist("images")
        image_records = []

        for image in images:
            # Determine if file is a video
            file_type = "video" if image.content_type.startswith("video/") else "image"
            
            # Extract video metadata if video
            duration, frame_rate, total_frames = None, None, None
            if file_type == "video":
                duration, frame_rate, total_frames = self._get_video_metadata(image)
            image_record = ImageVideoModel.objects.create(
                image=image,
                is_label=is_label,
                project=project,
                original_filename=image.name,
                type=file_type,
                duration=duration,
                frame_rate=frame_rate,
                total_frames=total_frames,
            )
            image_records.append(image_record)

        serializer = self.get_serializer(
            image_records, many=True, context={"request": request}
        )
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=["get"])
    def folder_coordinates(self, request):
        folder_path = request.query_params.get("folder_path")
        if not folder_path:
            return Response(
                {"error": "folder_path query parameter is required"}, status=400
            )

        images = ImageVideoModel.objects.filter(folder_path=folder_path)
        all_coordinates = []
        for image in images:
            coordinates = image.coordinates.all()
            serializer = CoordinateSerializer(coordinates, many=True)
            all_coordinates.extend(serializer.data)

        return Response(all_coordinates)

    @action(detail=False, methods=["post"])
    def save_all_coordinates(self, request):
        all_coordinates = request.data
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
                    image = ImageVideoModel.objects.get(id=image_id)
                except ImageVideoModel.DoesNotExist:
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
            mask_input = np.array(prev_mask, dtype=np.float32)
            mask_input = cv2.resize(
                mask_input, (256, 256), interpolation=cv2.INTER_LINEAR
            )
            mask_input = np.expand_dims(mask_input, axis=0)
            mask_input = mask_input.astype(np.uint8)

        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            mask_input=mask_input if mask_input is not None else None,
            multimask_output=False,
        )

        binary_mask = masks[0].astype(np.uint8)

        # Get complexity parameter
        complexity = request.query_params.get("complexity", 50)
        try:
            complexity = float(complexity)
            complexity = max(0, min(complexity, 100))  # Ensure it's within [0, 100]
        except ValueError:
            complexity = 50  # Default value if conversion fails

        # Generate polygons from mask using the provided method
        mask = binary_mask.astype(np.uint8)
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1)
        )
        contours = contours[0] if len(contours) == 2 else contours[1]

        polygons = []
        for contour in contours:
            arc_length = cv2.arcLength(contour, True)
            # Map complexity to epsilon
            epsilon = ((100 - complexity) / 100.0) * 0.1 * arc_length + (
                complexity / 100.0
            ) * 0.001 * arc_length
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Convert to list of (x, y) coordinates
            polygon = approx[:, 0, :].tolist()
            polygons.append(polygon)

        return Response({"mask": binary_mask.tolist(), "polygons": polygons})

    @action(detail=False, methods=["post"])
    def generate_polygon(self, request):
        data = request.data
        mask = data.get("mask")
        if mask is None:
            return Response({"error": "No mask provided."}, status=400)

        complexity = request.query_params.get("complexity", 50)
        try:
            complexity = float(complexity)
            complexity = max(0, min(complexity, 100))  # Ensure it's within [0, 100]
        except ValueError:
            complexity = 50  # Default value if conversion fails

        # Convert mask to numpy array
        binary_mask = np.array(mask, dtype=np.uint8)
        if len(binary_mask.shape) != 2:
            return Response({"error": "Invalid mask format."}, status=400)

        # Threshold the mask to ensure binary
        _, binary_mask = cv2.threshold(binary_mask, 0.5, 1, cv2.THRESH_BINARY)

        # Generate polygons from mask using the provided method
        mask = binary_mask.astype(np.uint8)
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1)
        )
        contours = contours[0] if len(contours) == 2 else contours[1]

        polygons = []
        for contour in contours:
            arc_length = cv2.arcLength(contour, True)
            # Map complexity to epsilon
            epsilon = ((100 - complexity) / 100.0) * 0.1 * arc_length + (
                complexity / 100.0
            ) * 0.001 * arc_length
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Convert to list of (x, y) coordinates
            polygon = approx[:, 0, :].tolist()
            polygons.append(polygon)

        return Response({"polygons": polygons})

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
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

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
