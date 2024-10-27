from django.db import models


class Coordinate(models.Model):
    folder_path = models.CharField(max_length=255)
    image_name = models.CharField(max_length=255)
    x = models.FloatField()
    y = models.FloatField()

    class Meta:
        unique_together = ("folder_path", "image_name")
