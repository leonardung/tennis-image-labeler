from io import BytesIO
from celery import shared_task, Celery
from ml_models.tennis_ball_detection.inter_on_video import process_images
import os
from .models.models import Coordinate  # Import the model here to avoid circular imports
import base64


@shared_task
def process_images_task(encoded_image_files, image_names, folder_path):
    coordinates_dict = {}

    # Decode images
    image_files = [
        BytesIO(base64.b64decode(encoded_file)) for encoded_file in encoded_image_files
    ]

    # Process images
    coordinates = process_images(image_files)

    # Create the result dictionary and update the database
    for image_name, coordinate in zip(image_names, coordinates):
        if coordinate[0] and coordinate[1]:
            coordinates_dict[image_name] = {"x": coordinate[0], "y": coordinate[1]}

            # Save to the database
            Coordinate.objects.update_or_create(
                folder_path=folder_path,
                image_name=image_name,
                defaults={"x": coordinate[0], "y": coordinate[1]},
            )
            
    return coordinates_dict
