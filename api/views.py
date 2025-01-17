import os
import cv2
from django.shortcuts import get_object_or_404
import numpy as np
import torch
import io
from PIL import Image
import tempfile
from django.conf import settings
from django.db import transaction
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from segment_anything import SamPredictor, sam_model_registry
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from api.models.coordinate import Coordinate
from api.models.image import ImageModel
from api.models.project import Project
from api.serializers import (
    CoordinateSerializer,
    ProjectSerializer,
    ImageModelSerializer,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.core.files.base import ContentFile

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
    queryset = ImageModel.objects.all()
    serializer_class = ImageModelSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    sam_model: SAM2VideoPredictor = None
    inference_state = None

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    def get_queryset(self):
        return ImageModel.objects.filter(project__user=self.request.user)

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
            image_record = ImageModel.objects.create(
                image=image,
                is_label=is_label,
                project=project,
                original_filename=image.name,
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

        images = ImageModel.objects.filter(folder_path=folder_path)
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
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        image: ImageModel = self.get_object()
        if self.__class__.sam_model is None:
            self.load_model()
            self.__class__.inference_state = self.__class__.sam_model.init_state(
                video_path=f"{settings.MEDIA_ROOT}/projects/{image.project.pk}/images"
            )
        data = request.data
        coordinates = data.get("coordinates")
        input_points = np.array([[coord["x"], coord["y"]] for coord in coordinates])
        input_labels = np.array(
            [1 if coord.get("include", True) else 0 for coord in coordinates]
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = (
                self.__class__.sam_model.add_new_points_or_box(
                    inference_state=self.__class__.inference_state,
                    frame_idx=int(image.image.name.split(".jpg")[0].split("/")[-1]),
                    obj_id=0,
                    points=input_points,
                    labels=input_labels,
                )
            )
        binary_mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
        mask_image = Image.fromarray((binary_mask * 255).astype(np.int8), mode="L")
        temp_buffer = io.BytesIO()
        mask_image.save(temp_buffer, format="PNG")
        temp_buffer.seek(0)
        if image.mask:
            image.mask.delete(save=False)
        image.mask.save(
            image.image.name.split(".jpg")[0].split("/")[-1] + ".png",
            ContentFile(temp_buffer.read()),
            save=True,  # This will write to disk and update the model
        )
        self.propagate_mask(request)

        return Response({"mask": binary_mask.astype(np.int8).tolist()})

    @action(detail=True, methods=["post"])
    def propagate_mask(self, request, pk=None):
        current_image = self.get_object()
        project = current_image.project
        project_images = ImageModel.objects.filter(project=project)
        video_segments = {}
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.sam_model.propagate_in_video(
                self.inference_state, start_frame_idx=0
            ):
                masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0)
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.int8)
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                video_segments[out_frame_idx] = masks
                matching_image = project_images.filter(
                    image__endswith=f"{out_frame_idx:05d}.jpg"
                ).first()
                if not matching_image:
                    continue
                for i, out_obj_id in enumerate(out_obj_ids):
                    binary_mask = (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    mask_image = Image.fromarray(
                        (binary_mask * 255).astype(np.uint8), mode="L"
                    )
                    temp_buffer = io.BytesIO()
                    mask_image.save(temp_buffer, format="PNG")
                    temp_buffer.seek(0)
                    # filename = f"mask_{out_frame_idx:05d}.png"
                    # if len(out_obj_ids) > 1:
                    #     filename = f"mask_{out_frame_idx:05d}_{out_obj_id}.png"
                    if matching_image.mask:
                        matching_image.mask.delete(save=False)
                    matching_image.mask.save(
                        matching_image.image.name.split(".jpg")[0].split("/")[-1] + ".png",
                        ContentFile(temp_buffer.read()),
                        save=True,  
                    )

        return Response({"detail": "All masks have been propagated and saved in one pass."})


    def load_model(self):
        if self.__class__.sam_model is None:
            try:
                sam_checkpoint = "./ml_models/sam_models/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                self.__class__.sam_model = build_sam2_video_predictor(
                    model_cfg, sam_checkpoint
                )
            except Exception as e:
                raise e


class VideoViewSet(viewsets.ViewSet):
    """
    Handles video uploads. Extracts frames with configurable stride and maximum frames,
    saves them to ImageVideoModel, and returns the created frame records.
    """

    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def create(self, request, *args, **kwargs):
        project_id = request.data.get("project_id")
        is_label = request.data.get("is_label", False)
        stride = int(request.data.get("stride", 1))
        max_frames = int(request.data.get("max_frames", 500))

        project = get_object_or_404(Project, id=project_id, user=request.user)

        # Expect a single "video" file
        video_file = request.FILES.get("video")
        if not video_file:
            return Response(
                {"error": "No video file found in the request."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not video_file.content_type.startswith("video/"):
            return Response(
                {"error": "Uploaded file is not a video."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 1) Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            for chunk in video_file.chunks():
                temp.write(chunk)
            temp_name = temp.name

        # 2) Extract frames using OpenCV
        cap = cv2.VideoCapture(temp_name)
        frame_records = []
        frame_index = 0
        saved_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # no more frames

            # Skip frames based on stride
            if frame_index % stride != 0:
                frame_index += 1
                continue

            # Stop if max_frames limit is reached
            if saved_frames >= max_frames:
                break

            # Convert the OpenCV frame to bytes
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                frame_index += 1
                continue

            frame_name = f"{saved_frames:05d}.jpg"
            frame_content = ContentFile(buffer.tobytes(), name=frame_name)

            frame_model = ImageModel.objects.create(
                image=frame_content,
                is_label=is_label,
                project=project,
                original_filename=frame_name,
            )
            frame_records.append(frame_model)
            saved_frames += 1
            frame_index += 1

        cap.release()

        serializer = ImageModelSerializer(
            frame_records, many=True, context={"request": request}
        )
        return Response(serializer.data, status=status.HTTP_201_CREATED)


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
