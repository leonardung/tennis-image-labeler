from django.core.management.base import BaseCommand
from django.contrib.auth.models import User


class Command(BaseCommand):
    help = "Seed a test user"

    def handle(self, *args, **kwargs):
        username = "testuser"
        password = "testpassword"
        email = "testuser@example.com"

        if User.objects.filter(username=username).exists():
            self.stdout.write(self.style.WARNING(f"User '{username}' already exists."))
        else:
            User.objects.create_user(username=username, email=email, password=password)
            self.stdout.write(self.style.SUCCESS(f"User '{username}' created with password '{password}'."))
