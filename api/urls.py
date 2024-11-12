from django.urls import include, path
from rest_framework.routers import DefaultRouter
from api.views import ImageViewSet, ModelManagerViewSet

router = DefaultRouter()
router.register(r"images", ImageViewSet)
router.register(r"model", ModelManagerViewSet, basename="model")

urlpatterns = [
    path("", include(router.urls)),
]
