# api/views.py
import json
import urllib
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.core.handlers.wsgi import WSGIRequest
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from celery.result import AsyncResult

from api.models import Coordinate
from ml_models.tennis_ball_detection.inter_on_video import process_images

from .tasks import process_images_task


# from .tasks import process_images_task


@csrf_exempt
def get_coordinates(request):
    if request.method == "GET":
        folder_path = request.GET.get("folder_path")
        if not folder_path:
            return HttpResponseBadRequest("Missing folder_path parameter")

        coordinates = Coordinate.objects.filter(folder_path=folder_path)
        coordinates_data = {
            coordinate.image_name: {"x": coordinate.x, "y": coordinate.y}
            for coordinate in coordinates
        }
        return JsonResponse({"coordinates": coordinates_data})
    else:
        return HttpResponseBadRequest("Invalid request method")


@csrf_exempt
def save_coordinates(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            folder_path = data.get("folder_path")
            image_name = data.get("image_name")
            x = data.get("x")
            y = data.get("y")

            if not folder_path or not image_name or x is None or y is None:
                return HttpResponseBadRequest("Missing required fields")

            # Update or create the coordinate in the database
            Coordinate.objects.update_or_create(
                folder_path=folder_path,
                image_name=image_name,
                defaults={"x": x, "y": y},
            )

            return JsonResponse({"status": "success"})
        except json.JSONDecodeError:
            return HttpResponseBadRequest("Invalid JSON data")
    else:
        return HttpResponseBadRequest("Invalid request method")


@csrf_exempt
def calculate_coordinates_(request: WSGIRequest):
    if request.method == "POST":
        images = request.FILES.getlist("images")
        coordinates_dict = {}

        coordinates = process_images(images)

        for file, coordinate in zip(images, coordinates):
            image_name = file.name
            if coordinate[0] and coordinate[1]:
                coordinates_dict[image_name] = {"x": coordinate[0], "y": coordinate[1]}

        return JsonResponse({"coordinates": coordinates_dict})
    return JsonResponse({"status": "error"}, status=400)


@csrf_exempt
def calculate_coordinates(request: WSGIRequest):
    if request.method == "POST":
        images = request.FILES.getlist("images")
        folder_path = request.POST.get("folder_path")
        saved_image_paths = []

        for file in images:
            file_path = default_storage.save(
                f"temp_images/{file.name}", ContentFile(file.read())
            )
            saved_image_paths.append(file_path)

        task = process_images_task.delay(saved_image_paths, folder_path)

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
