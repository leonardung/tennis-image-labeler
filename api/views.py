# api/views.py
import base64
import json
import os
import urllib
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.core.handlers.wsgi import WSGIRequest
from celery.result import AsyncResult
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.db import transaction

from api.models import Coordinate, ImageModel
from api.serializers import CoordinateSerializer, ImageModelSerializer

from .tasks import process_images_task


@csrf_exempt
def upload_images(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    folder_path = request.POST.get("folder_path", "default_folder")
    images = request.FILES.getlist("images")
    image_data = []

    for image in images:
        relative_path = folder_path + "/" + image.name
        full_path = settings.MEDIA_ROOT + "/" + relative_path

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
        coordinates = coordinates[0] if coordinates else []  # only 1 coord for now
        image_data.append(
            {
                "id": image_record.id,
                "url": request.build_absolute_uri(settings.MEDIA_URL + relative_path),
                "filename": image.name,
                "coordinates": coordinates,
            }
        )

    return JsonResponse({"images": image_data})


@csrf_exempt
def get_coordinates(request):
    if request.method == "GET":
        folder_path = request.GET.get("folder_path")
        if not folder_path:
            return HttpResponseBadRequest("Missing folder_path parameter")

        # Retrieve all images in the specified folder
        images = ImageModel.objects.filter(folder_path=folder_path).prefetch_related(
            "coordinates"
        )

        # Construct the response data
        coordinates_data = {}
        for image in images:
            image_name = image.image.name.split("/")[-1].split("\\")[-1]
            coords = image.coordinates.all()
            if coords.exists():
                coordinates_data[image_name] = [
                    {"x": coord.x, "y": coord.y} for coord in coords
                ]

        return JsonResponse({"coordinates": coordinates_data})
    else:
        return HttpResponseBadRequest("Invalid request method")


@csrf_exempt
def save_coordinates(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    # Parse JSON data from the request body
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)

    coordinates_list = data.get("coordinates", [])
    if not coordinates_list:
        return JsonResponse({"error": "No coordinates provided"}, status=400)

    # Group coordinates by image to handle multiple coordinates per image
    coordinates_by_image = {}
    for coord_data in coordinates_list:
        folder_path = coord_data.get("folder_path")
        image_name = coord_data.get("image_name")
        x = coord_data.get("x")
        y = coord_data.get("y")

        if not folder_path or not image_name or x is None or y is None:
            return JsonResponse({"error": "Missing data in coordinates"}, status=400)

        image_key = (folder_path, image_name)
        if image_key not in coordinates_by_image:
            coordinates_by_image[image_key] = []

        coordinates_by_image[image_key].append({"x": x, "y": y})

    with transaction.atomic():
        for (folder_path, image_name), coords in coordinates_by_image.items():
            relative_path = os.path.join(folder_path, image_name)

            # Try to retrieve the ImageModel instance
            try:
                image = ImageModel.objects.get(image=relative_path)
            except ImageModel.DoesNotExist:
                return JsonResponse(
                    {"error": f"Image not found: {relative_path}"}, status=404
                )

            # Clear existing coordinates for this image
            image.coordinates.all().delete()

            # Create new Coordinate instances
            coordinate_objs = [
                Coordinate(image=image, x=coord["x"], y=coord["y"]) for coord in coords
            ]

            # Bulk create the coordinates for efficiency
            Coordinate.objects.bulk_create(coordinate_objs)

    return JsonResponse({"status": "success"}, status=200)


@csrf_exempt
def calculate_coordinates(request: WSGIRequest):
    if request.method == "POST":
        images = request.FILES.getlist("images")
        folder_path = request.POST.get("folder_path")
        image_files = []
        image_names = []

        for file in images:
            if isinstance(file, InMemoryUploadedFile):
                image_files.append(base64.b64encode(file.read()).decode("utf-8"))
                image_names.append(file.name)
        task = process_images_task.delay(image_files, image_names, folder_path)

        return JsonResponse({"task_id": task.id})
    return JsonResponse({"status": "error"}, status=400)


@csrf_exempt
def check_celery_task_status(request, pk: str):
    """Check status of a running Celery task"""
    task_id = urllib.parse.unquote(pk)
    task = AsyncResult(task_id)

    if task.ready():
        if task.successful():
            result = task.result
            return JsonResponse(
                {
                    "status": "success",
                    "result": result,
                }
            )
        elif task.failed():
            return JsonResponse(
                {
                    "status": "failed",
                    "error": str(task.result),  # task.result contains the exception
                }
            )
    else:
        return JsonResponse({"status": "pending"})
