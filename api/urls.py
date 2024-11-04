# api/urls.py
from django.urls import path
from .views import (
    check_celery_task_status,
    calculate_coordinates,
    get_coordinates,
    save_coordinates,
    upload_images,
)

urlpatterns = [
    path("calculate-coordinates/", calculate_coordinates, name="calculate-coordinates"),
    path("get-coordinates/", get_coordinates, name="get-coordinates"),
    path("save-coordinates/", save_coordinates, name="save-coordinates"),
    path("upload-images/", upload_images, name="upload_images"),
]
