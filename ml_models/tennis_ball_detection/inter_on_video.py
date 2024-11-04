from PIL import Image
from .model import BallTrackerNet
import torch
import cv2
from .general import postprocess
from tqdm import tqdm
import numpy as np
import argparse
from scipy.spatial import distance


def process_frame(frame, model, device, ball_track, dists):
    """Process a single frame
    :params
        frame: current frame
        model: pretrained model
        device: device to run the model on
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """
    height = 360
    width = 640
    run_inference = True
    img = cv2.resize(frame, (width, height))
    if ball_track[-1][1] is not None:
        img_prev = cv2.resize(ball_track[-1][1], (width, height))
    else:
        np.zeros((height, width, 3))
        run_inference = False
    if ball_track[-2][1] is not None:
        img_preprev = cv2.resize(ball_track[-2][1], (width, height))
    else:
        np.zeros((height, width, 3))
        run_inference = False

    if run_inference:
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)
        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)
    else:
        x_pred, y_pred = None, None
    ball_track.append(((x_pred, y_pred), frame))

    if ball_track[-1][0][0] and ball_track[-2][0][0]:
        dist = distance.euclidean(ball_track[-1][0], ball_track[-2][0])
    else:
        dist = -1
    dists.append(dist)
    return ball_track, dists


def read_and_process_video(path_video, model, device, output_video_path, fps):
    """Read and process video file frame by frame to reduce memory consumption
    :params
        path_video: path to video file
        model: pretrained model
        device: device to run the model on
        output_video_path: path to output video file
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ball_track = [((None, None), None)] * 2
    dists = [-1] * 2

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, (1280, 720)
    )

    j = 0
    with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            if j > 100:
                break
            j += 1
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1280, 720))
            ball_track, dists = process_frame(frame, model, device, ball_track, dists)
            frame_num = len(ball_track) - 1
            for i in range(1):
                if frame_num - i > 0:
                    if ball_track[frame_num - i][0][0]:
                        x = int(ball_track[frame_num - i][0][0])
                        y = int(ball_track[frame_num - i][0][1])
                        frame = cv2.circle(
                            frame,
                            (x, y),
                            radius=0,
                            color=(0, 0, 255),
                            thickness=max(10 - i, 1),
                        )
                    else:
                        break
            out.write(frame)
            pbar.update(1)

            if len(ball_track) > 7:
                del ball_track[0]
                del dists[0]

    cap.release()
    out.release()


async def process_images(images):
    """Process a list of PIL images and return a list of coordinates (x, y)
    :params
        images: list of PIL images
        model: pretrained model
        device: device to run the model on
    :return:
        List of (x, y) coordinates for detected points in each image
    """
    # Convert images from PIL to OpenCV format
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

    with tqdm(total=len(images), desc="Processing Images", unit="image") as pbar:
        for image_file in images:
            image = cv2.cvtColor(
                np.array(Image.open(image_file.image.path)), cv2.COLOR_RGB2BGR
            )

            # Resize the image if needed
            image = cv2.resize(image, (1280, 720))
            # Process the frame and update ball_track and dists
            ball_track, dists = process_frame(image, model, device, ball_track, dists)
            frame_num = len(ball_track) - 1
            # Extract coordinates of detected points
            for i in range(1):
                if frame_num - i > 0 and ball_track[frame_num - i][0][0]:
                    x = int(ball_track[frame_num - i][0][0])
                    y = int(ball_track[frame_num - i][0][1])
                    yield image_file, (x, y)
                else:
                    yield image_file, (None, None)
            pbar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/2024-06-18-20-02-28/best_epoch.pth",
        help="path to model",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="50 Tennis Shots.mp4",
        help="path to input video",
    )
    parser.add_argument(
        "--video_out_path",
        type=str,
        default="50 Tennis Shots TRACKED.avi",
        help="path to output video",
    )
    parser.add_argument(
        "--extrapolation",
        default=False,
        action="store_true",
        help="whether to use ball track extrapolation",
    )
    args = parser.parse_args()

    model = BallTrackerNet()
    device = "cuda"
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    read_and_process_video(args.video_path, model, device, args.video_out_path, fps)
