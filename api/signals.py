# myapp/signals.py
import os
import shutil
from django.conf import settings
from django.db.models.signals import post_delete
from django.dispatch import receiver
from .models import ImageModel, Project


@receiver(post_delete, sender=ImageModel)
def auto_delete_file_on_delete(sender, instance: ImageModel, **kwargs):
    """
    Deletes image and thumbnail files from the filesystem
    when the corresponding `ImageModel` object is deleted.
    """
    # If the image field was filled, delete the file
    if instance.image:
        instance.image.delete(save=False)

    # If the thumbnail field was filled, delete the file
    if instance.thumbnail:
        instance.thumbnail.delete(save=False)


@receiver(post_delete, sender=Project)
def delete_project_folder(sender, instance: Project, **kwargs):
    """
    Removes the entire folder projects/<project_id> from the filesystem
    when the corresponding `Project` object is deleted.
    """
    project_folder = os.path.join(settings.MEDIA_ROOT, "projects", str(instance.id))
    if os.path.exists(project_folder):
        shutil.rmtree(project_folder, ignore_errors=True)
