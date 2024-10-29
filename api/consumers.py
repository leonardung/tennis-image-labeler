from channels.generic.websocket import AsyncWebsocketConsumer
import json
import base64
from io import BytesIO
from channels.db import database_sync_to_async

from api.models import Coordinate
from ml_models.tennis_ball_detection.inter_on_video import process_images


class ImageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            data = json.loads(text_data)
            images_data = data.get('images', [])
            folder_path = data.get('folder_path', '')
            image_files = []
            image_names = []

            for image_dict in images_data:
                image_name = image_dict['name']
                image_content = image_dict['content']
                image_bytes = base64.b64decode(image_content)
                image_files.append(BytesIO(image_bytes))
                image_names.append(image_name)

            # Process images and save results asynchronously
            coordinates_dict = await self.process_images_and_save(
                image_files, image_names, folder_path
            )

            # Send back the result via WebSocket
            await self.send(text_data=json.dumps({
                'status': 'success',
                'coordinates': coordinates_dict,
            }))
        else:
            await self.send(text_data=json.dumps({'status': 'error', 'message': 'No data received'}))

    @database_sync_to_async
    def process_images_and_save(self, image_files, image_names, folder_path):
        coordinates = process_images(image_files)
        coordinates_dict = {}

        for image_name, coordinate in zip(image_names, coordinates):
            if coordinate[0] and coordinate[1]:
                coordinates_dict[image_name] = {"x": coordinate[0], "y": coordinate[1]}
                Coordinate.objects.update_or_create(
                    folder_path=folder_path,
                    image_name=image_name,
                    defaults={"x": coordinate[0], "y": coordinate[1]},
                )

        return coordinates_dict
