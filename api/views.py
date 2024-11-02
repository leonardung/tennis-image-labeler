# api/views.py
import json
import base64
import os
import urllib
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.core.handlers.wsgi import WSGIRequest
from celery.result import AsyncResult
from django.core.files.uploadedfile import InMemoryUploadedFile

from api.models import Coordinate, ImageModel
from ml_models.tennis_ball_detection.inter_on_video import process_images

from .tasks import process_images_task


@csrf_exempt
def upload_images(request):
    if request.method == "POST":
        folder_path = request.POST.get("folder_path", "default_folder")
        images = request.FILES.getlist("images")
        image_urls = []
        coordinates = {}  # Implement logic to get coordinates if needed

        for image in images:
            # Construct file path within media directory
            relative_path = folder_path + "/" + image.name
            full_path = os.path.join(settings.MEDIA_ROOT, relative_path)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Save the file
            with open(full_path, "wb+") as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Save a reference in the database
            image_record = ImageModel.objects.create(
                image=relative_path,
                folder_path=folder_path,
            )

            # Collect the image URL
            image_url = request.build_absolute_uri(settings.MEDIA_URL + relative_path)
            image_urls.append(image_url)

        # Return a JSON response with image URLs and coordinates
        return JsonResponse(
            {
                "image_urls": image_urls,
                "coordinates": coordinates,
            }
        )
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)


# def upload_images(request):
#     if request.method == "POST":
#         folder_path = request.POST.get("folder_path", "default_folder")
#         images = request.FILES.getlist("images")
#         image_data = []

#         for image in images:
#             # ... [Saving image code as before] ...

#             # Assume you have logic to get coordinates for each image
#             coordinates = get_coordinates_for_image(image_record)

#             image_data.append(
#                 {
#                     "url": request.build_absolute_uri(
#                         settings.MEDIA_URL + relative_path
#                     ),
#                     "filename": image.name,
#                     "coordinates": coordinates,
#                 }
#             )

#         return JsonResponse({"images": image_data})
#     else:
#         return JsonResponse({"error": "Invalid request method"}, status=400)


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
            coordinates_list = data.get("coordinates", [])

            if not coordinates_list:
                return HttpResponseBadRequest("Missing coordinates data")

            for coordinate in coordinates_list:
                folder_path = coordinate.get("folder_path")
                image_name = coordinate.get("image_name")
                x = coordinate.get("x")
                y = coordinate.get("y")

                if not folder_path or not image_name or x is None or y is None:
                    return HttpResponseBadRequest(
                        "Missing required fields in coordinate"
                    )

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
        print(image_names)

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
