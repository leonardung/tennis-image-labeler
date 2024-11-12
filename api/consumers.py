from channels.generic.websocket import AsyncWebsocketConsumer
import json
from asgiref.sync import sync_to_async
import cv2
import torch
from api.models import Coordinate, ImageModel
from ml_models.tennis_ball_detection.model import BallTrackerNet
from ml_models.tennis_ball_detection.inter_on_video import process_frame

class ImageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("WebSocket connection accepted.")

    async def disconnect(self, close_code):
        print(f"WebSocket disconnected with code {close_code}.")

    async def receive(self, text_data):
        data = json.loads(text_data)
        folder_path = data.get("folder_path")

        if folder_path:
            # Fetch image records asynchronously
            image_records = await sync_to_async(
                lambda: list(ImageModel.objects.filter(folder_path=folder_path))
            )()
            print(image_records[0].pk)
            print(image_records[0].id)

            if not image_records:
                await self.send(text_data=json.dumps({
                    "status": "error",
                    "message": f"No images found in folder: {folder_path}"
                }))
                return

            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = BallTrackerNet()
            model_path = "ml_models/tennis_ball_detection/best_epoch.pth"
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device)
                model.eval()
            except Exception as e:
                await self.send(text_data=json.dumps({
                    "status": "error",
                    "message": f"Failed to load model: {str(e)}"
                }))
                return

            ball_track = [((None, None), None)] * 2
            dists = [-1] * 2
            total_images = len(image_records)
            processed_images = 0

            for image_file in image_records:
                try:
                    image = cv2.imread(image_file.image.path)
                    if image is None:
                        raise ValueError("Failed to read image.")
                    image = cv2.resize(image, (1280, 720))
                except Exception as e:
                    await self.send(text_data=json.dumps({
                        "status": "error",
                        "message": f"Error processing image {image_file.image.name}: {str(e)}"
                    }))
                    continue  # Skip to the next image

                ball_track, dists = process_frame(
                    image, model, device, ball_track, dists
                )
                frame_num = len(ball_track) - 1
                coordinates = (None, None)
                for i in range(1):
                    if frame_num - i > 0 and ball_track[frame_num - i][0][0]:
                        x = int(ball_track[frame_num - i][0][0])
                        y = int(ball_track[frame_num - i][0][1])
                        coordinates = (x, y)
                    else:
                        coordinates = (None, None)

                if coordinates[0] is not None and coordinates[1] is not None:
                    # Update or create the coordinate in the database
                    await sync_to_async(Coordinate.objects.update_or_create)(
                        image=image_file,
                        defaults={'x': coordinates[0], 'y': coordinates[1]}
                    )

                processed_images += 1
                progress = (processed_images / total_images) * 100

                # Send progress update to the frontend
                await self.send(text_data=json.dumps({
                    "status": "success",
                    "coordinates": {
                        "image_id": image_file.id,
                        "x": coordinates[0],
                        "y": coordinates[1],
                    },
                    "progress": progress,
                }))

            await self.send(text_data=json.dumps({"status": "complete"}))
        else:
            await self.send(text_data=json.dumps({
                "status": "error",
                "message": "Missing folder_path parameter."
            }))