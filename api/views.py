# api/views.py
import io
import random
import PIL.Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.handlers.wsgi import WSGIRequest
import json
import os
import PIL
import numpy as np
import torch

from ml_models.tennis_ball_detection.inter_on_video import process_images
from ml_models.tennis_ball_detection.model import BallTrackerNet


@csrf_exempt
def get_coordinates(request: WSGIRequest):
    if request.method == "POST":
        buffer = io.BytesIO()
        for chunk in request:
            buffer.write(chunk)
        buffer.seek(0)
        images = split_images(buffer)

        coordinates_dict = {}
        model = BallTrackerNet()
        device = "cuda"
        model.load_state_dict(
            torch.load(
                r"ml_models\tennis_ball_detection\best_epoch.pth",
                map_location=device,
            )
        )
        model = model.to(device)
        model.eval()
        coordinates = process_images(images, model, device)

        for file, coordinate in zip(images, coordinates):
            image_name = file.name
            if coordinate[0] and coordinate[1]:
                coordinates_dict[image_name] = {"x": coordinate[0], "y": coordinate[1]}

        return JsonResponse({"coordinates": coordinates_dict})
    return JsonResponse({"status": "error"}, status=400)


def split_images(buffer):
    """
    Split the buffer into separate images based on a custom delimiter.
    Example assumes that each image is separated by a specific byte sequence.
    """
    delimiter = b"--image-separator--"  # Custom delimiter to separate images
    image_chunks = buffer.getvalue().split(delimiter)

    images = []
    for image_chunk in image_chunks:
        if image_chunk:
            image_io = io.BytesIO(image_chunk)
            images.append(image_io)

    return images
