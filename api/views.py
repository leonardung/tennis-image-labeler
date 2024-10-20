# api/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.handlers.wsgi import WSGIRequest
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .tasks import process_images_task

# from .tasks import process_images_task


# @csrf_exempt
# def get_coordinates(request: WSGIRequest):
#     if request.method == "POST":
#         images = request.FILES.getlist("images")
#         coordinates_dict = {}

#         coordinates = process_images(images)

#         for file, coordinate in zip(images, coordinates):
#             image_name = file.name
#             if coordinate[0] and coordinate[1]:
#                 coordinates_dict[image_name] = {"x": coordinate[0], "y": coordinate[1]}

#         return JsonResponse({"coordinates": coordinates_dict})
#     return JsonResponse({"status": "error"}, status=400)


@csrf_exempt
def get_coordinates(request: WSGIRequest):
    if request.method == "POST":
        images = request.FILES.getlist("images")
        saved_image_paths = []

        # Save images temporarily
        for file in images:
            file_path = default_storage.save(
                f"temp_images/{file.name}", ContentFile(file.read())
            )
            saved_image_paths.append(file_path)
        print(len(saved_image_paths))
        print(saved_image_paths)

        # Push the task to the queue
        task = process_images_task.delay(saved_image_paths)

        # Return the task ID to the client
        return JsonResponse({"task_id": task.id})

    return JsonResponse({"status": "error"}, status=400)


from celery.result import AsyncResult
from django.core.cache import cache


@csrf_exempt
def check_task_status(request, task_id):
    result = AsyncResult(task_id)

    if result.state == "PENDING":
        response = {"status": "pending"}
    elif result.state == "SUCCESS":
        # Retrieve the result from the cache
        coordinates_dict = cache.get(f"task_result_{task_id}")
        response = {"status": "success", "coordinates": coordinates_dict}
    elif result.state == "FAILURE":
        response = {"status": "failed", "error": str(result.info)}
    else:
        response = {"status": result.state}

    return JsonResponse(response)
