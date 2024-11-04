from channels.generic.websocket import AsyncWebsocketConsumer
import json
from asgiref.sync import sync_to_async
import cv2
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm
from api.models import Coordinate, ImageModel
from ml_models.tennis_ball_detection.model import BallTrackerNet
from ml_models.tennis_ball_detection.inter_on_video import process_frame


class ImageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        folder_path = data.get("folder_path")

        if folder_path:
            image_records = await sync_to_async(
                lambda: list(ImageModel.objects.filter(folder_path=folder_path))
            )()
            model = BallTrackerNet()
            device = "cuda"
            model.load_state_dict(
                torch.load(
                    "ml_models/tennis_ball_detection/best_epoch.pth",
                    map_location=device,
                )
            )
            model = model.to(device)
            model.eval()
            ball_track = [((None, None), None)] * 2
            dists = [-1] * 2
            for image_file in tqdm(image_records):
                image = cv2.cvtColor(
                    np.array(Image.open(image_file.image.path)), cv2.COLOR_RGB2BGR
                )

                image = cv2.resize(image, (1280, 720))
                ball_track, dists = process_frame(
                    image, model, device, ball_track, dists
                )
                frame_num = len(ball_track) - 1
                for i in range(1):
                    if frame_num - i > 0 and ball_track[frame_num - i][0][0]:
                        x = int(ball_track[frame_num - i][0][0])
                        y = int(ball_track[frame_num - i][0][1])
                        coordinates = (x, y)
                    else:
                        coordinates = (None, None)
                if coordinates[0] is not None and coordinates[1] is not None:
                    await sync_to_async(Coordinate.objects.update_or_create)(
                        image=image_file, x=coordinates[0], y=coordinates[1]
                    )
                await self.send(
                    text_data=json.dumps(
                        {
                            "status": "success",
                            "coordinates": {
                                "image_name": image_file.image.name.split("/")[-1],
                                "x": coordinates[0],
                                "y": coordinates[1],
                            },
                            "processed_image_id": image_file.id,
                        }
                    )
                )
            await self.send(text_data=json.dumps({"status": "complete"}))
