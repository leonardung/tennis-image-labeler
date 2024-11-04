from django.db import models


class ImageModel(models.Model):
    image = models.ImageField(upload_to="images/")
    folder_path = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

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
