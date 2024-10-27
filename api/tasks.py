from celery import shared_task
from ml_models.tennis_ball_detection.inter_on_video import process_images
import os
from .models import Coordinate  # Import the model here to avoid circular imports


@shared_task
def process_images_task(image_paths, folder_path):

    coordinates_dict = {}

    # Process images
    coordinates = process_images([open(path, "rb") for path in image_paths])

    # Create the result dictionary and update the database
    for file_path, coordinate in zip(image_paths, coordinates):
        image_name = os.path.basename(file_path)

        if coordinate[0] and coordinate[1]:
            coordinates_dict[image_name] = {"x": coordinate[0], "y": coordinate[1]}

            # Save to the database
            Coordinate.objects.update_or_create(
                folder_path=folder_path,
                image_name=image_name,
                defaults={"x": coordinate[0], "y": coordinate[1]},
            )
    return coordinates_dict
