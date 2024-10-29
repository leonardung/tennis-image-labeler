from django.urls import path
from api.consumers import ImageConsumer

websocket_urlpatterns = [
    path('ws/process-images/', ImageConsumer.as_asgi()),
]