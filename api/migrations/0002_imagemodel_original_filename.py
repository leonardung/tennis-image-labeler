# Generated by Django 5.1.2 on 2024-11-23 13:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="imagemodel",
            name="original_filename",
            field=models.CharField(blank=True, max_length=255),
        ),
    ]
