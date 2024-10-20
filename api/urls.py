# api/urls.py
from django.urls import path
from .views import check_task_status, get_coordinates

urlpatterns = [
    path("get-coordinates/", get_coordinates, name="get-coordinates"),
    path('check-task-status/<str:task_id>/', check_task_status, name='check_task_status'),
]
