# api/urls.py
from django.urls import path
from .views import get_coordinates

urlpatterns = [
    path("get-coordinates/", get_coordinates, name="get-coordinates"),
]
