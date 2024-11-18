from django.urls import include, path
from rest_framework.routers import DefaultRouter
from api.views import ImageViewSet, ModelManagerViewSet, ProjectViewSet

router = DefaultRouter()
router.register(r"projects", ProjectViewSet, basename="project")
router.register(r"images", ImageViewSet, basename="images")
router.register(r"model", ModelManagerViewSet, basename="model")

urlpatterns = [
    path("", include(router.urls)),
]
