from django.db import models

from api.models.image import ImageModel


class Coordinate(models.Model):
    image = models.ForeignKey(
        ImageModel,
        related_name="coordinates",
        on_delete=models.CASCADE,
    )
    x = models.FloatField()
    y = models.FloatField()
    include = models.BooleanField(default=False)

    def __str__(self):
        return f"Coordinate ({self.x}, {self.y}) for {self.image}"
