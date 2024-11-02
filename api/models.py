from django.db import models


class Coordinate(models.Model):
    folder_path = models.CharField(max_length=255)
    image_name = models.CharField(max_length=255)
    x = models.FloatField()
    y = models.FloatField()

    class Meta:
        unique_together = ("folder_path", "image_name")


class ImageModel(models.Model):
    image = models.ImageField(upload_to="images/")  # The path is managed in the view
    folder_path = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image.name
