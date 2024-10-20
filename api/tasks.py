from celery import shared_task
from ml_models.tennis_ball_detection.inter_on_video import process_images
import os


@shared_task
def process_images_task(image_paths):

    coordinates_dict = {}

    # Process images
    coordinates = process_images([open(path, "rb") for path in image_paths])

    # Create the result dictionary
    for file_path, coordinate in zip(image_paths, coordinates):
        image_name = os.path.basename(file_path)
        if coordinate[0] and coordinate[1]:
            coordinates_dict[image_name] = {"x": coordinate[0], "y": coordinate[1]}

    # Cleanup: remove saved images after processing
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    # Store the result in the cache or database
    from django.core.cache import cache

    cache.set(
        f"task_result_{process_images_task.request.id}", coordinates_dict, timeout=3600
    )

    return coordinates_dict
