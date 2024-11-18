from django.db import models
from django.contrib.auth.models import User  # Assuming you're using Django's default User model


class Project(models.Model):
    PROJECT_TYPE_CHOICES = [
        ('point_coordinate', 'Point Coordinate'),
        ('bounding_box', 'Bounding Box'),
        ('segmentation', 'Segmentation'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='projects')
    name = models.CharField(max_length=255)
    type = models.CharField(max_length=20, choices=PROJECT_TYPE_CHOICES)

    def __str__(self):
        return f"{self.name} ({self.type})"


class ImageModel(models.Model):
    image = models.ImageField(upload_to="images/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_label = models.BooleanField(default=False)
    project = models.ForeignKey(Project, related_name='images', on_delete=models.CASCADE)

    def __str__(self):
        return self.image.name


class Coordinate(models.Model):
    image = models.ForeignKey(
        ImageModel,
        related_name="coordinates",
        on_delete=models.CASCADE,
    )
    x = models.FloatField()
    y = models.FloatField()

    def __str__(self):
        return f"Coordinate ({self.x}, {self.y}) for {self.image}"

    class Meta:
        unique_together = ("image", "x", "y")
