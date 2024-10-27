# api/urls.py
from django.urls import path, re_path
from .views import (
    check_celery_task_status,
    calculate_coordinates,
    get_coordinates,
    save_coordinates,
)

urlpatterns = [
    path("calculate-coordinates/", calculate_coordinates, name="calculate-coordinates"),
    path("get-coordinates/", get_coordinates, name="get-coordinates"),
    path("save-coordinates/", save_coordinates, name="save-coordinates"),
    # re_path(
    #     r"^check-task-status/(?P<pk>[^/]+)/$",
    #     check_celery_task_status,
    #     name="check_celery_task_status",
    # ),
]
