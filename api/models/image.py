import cv2
import numpy as np
import os
from io import BytesIO
from django.db import models
from django.core.files.base import ContentFile


def image_upload_path(instance, filename):
    return f"projects/{instance.project.id}/images/{filename}"


def thumbnail_upload_path(instance, filename):
    return f"projects/{instance.project.id}/thumbnails/{filename}"


def mask_upload_path(instance, filename):
    return f"projects/{instance.project.id}/masks/{filename}"


class ImageModel(models.Model):
    image = models.ImageField(upload_to=image_upload_path)
    thumbnail = models.ImageField(
        upload_to=thumbnail_upload_path, blank=True, null=True
    )
    mask = models.ImageField(upload_to=mask_upload_path, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_label = models.BooleanField(default=False)
    project = models.ForeignKey(
        "Project", related_name="images", on_delete=models.CASCADE
    )
    original_filename = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.image.name

    def save(self, *args, **kwargs):
        """
        Override save to generate a thumbnail when a new image is uploaded.
        """
        # Check if this is a new record or an update
        creating_new = self.pk is None

        super().save(*args, **kwargs)

        # If a new image is uploaded (and we don't already have a thumbnail), create it
        if creating_new and self.image and not self.thumbnail:
            self._create_thumbnail()
            # Save again to store the generated thumbnail
            super().save(update_fields=["thumbnail"])

    def _create_thumbnail(self):
        """
        Create a thumbnail using OpenCV while keeping aspect ratio.
        The longest side will be scaled down to 200px, if needed.
        """
        # Open and read the original image
        self.image.open()
        image_bytes = self.image.read()
        self.image.close()

        # Convert the image bytes into an OpenCV-compatible format
        np_arr = np.frombuffer(image_bytes, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_image is None:
            return  # Could not decode image; handle error as needed

        # Get the current size
        height, width = cv_image.shape[:2]
        max_dimension = 100

        # Determine the scale factor such that the larger side = 200 (if needed)
        max_side = max(width, height)
        if max_side > max_dimension:
            scale = max_dimension / float(max_side)
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Resize while keeping aspect ratio
            cv_thumb = cv2.resize(
                cv_image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
        else:
            # If the image is already small enough, don't upscale
            cv_thumb = cv_image

        # Encode the thumbnail back to JPEG
        success, encoded_image = cv2.imencode(".jpg", cv_thumb)
        if not success:
            return  # Encoding failed; handle error as needed

        # Use a BytesIO to create an in-memory file
        thumb_io = BytesIO(encoded_image.tobytes())

        # Build a thumbnail filename (e.g. "image_thumb.jpg")
        base, ext = os.path.splitext(os.path.basename(self.image.name))
        thumb_filename = f"{base}_thumb.jpg"

        # Save the thumbnail to the `thumbnail` field without calling save() again
        self.thumbnail.save(
            thumb_filename, ContentFile(thumb_io.getvalue()), save=False
        )
