from django.db import models
from django.contrib.auth.models import User


class Project(models.Model):
    PROJECT_TYPE_CHOICES = [
        ("point_coordinate", "Point Coordinate"),
        ("multi_point_coordinate", "Multi Point Coordinate"),
        ("bounding_box", "Bounding Box"),
        ("segmentation", "Segmentation"),
        ("video_tracking_segmentation", "Video Tracking Segmentation"),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="projects")
    name = models.CharField(max_length=255)
    type = models.CharField(max_length=30, choices=PROJECT_TYPE_CHOICES)

    def __str__(self):
        return f"{self.name} ({self.type})"


class ImageVideoModel(models.Model):
    image = models.ImageField(upload_to="images/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_label = models.BooleanField(default=False)
    project = models.ForeignKey(
        Project, related_name="images", on_delete=models.CASCADE
    )
    original_filename = models.CharField(max_length=255, blank=True)
    type = models.CharField(max_length=30, choices=[("image", "Image"), ("video", "Video")], default="image")
    duration = models.FloatField(null=True, blank=True)
    frame_rate = models.FloatField(null=True, blank=True)
    total_frames = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.image.name


class Coordinate(models.Model):
    image = models.ForeignKey(
        ImageVideoModel,
        related_name="coordinates",
        on_delete=models.CASCADE,
    )
    x = models.FloatField()
    y = models.FloatField()

    def __str__(self):
        return f"Coordinate ({self.x}, {self.y}) for {self.image}"

    class Meta:
        unique_together = ("image", "x", "y")
