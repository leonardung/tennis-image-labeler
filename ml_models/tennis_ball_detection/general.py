import torch
import time
import numpy as np
import torch.nn as nn
import cv2
from scipy.spatial import distance
from tqdm import tqdm


def train(model, train_loader, optimizer, device, epoch, max_iters=200):
    start_time = time.time()
    losses = []
    criterion = nn.CrossEntropyLoss()
    for iter_id, batch in enumerate(train_loader):
        optimizer.zero_grad()
        model.train()
        out = model(batch[0].float().to(device))
        gt = torch.tensor(batch[1], dtype=torch.long, device=device)
        loss = criterion(out, gt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end_time = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        print(
            "train | epoch = {}, iter = [{}|{}], loss = {}, time = {}".format(
                epoch, iter_id, max_iters, round(loss.item(), 6), duration
            )
        )
        losses.append(loss.item())

        if iter_id > max_iters - 1:
            break

    return np.mean(losses)


import matplotlib.pyplot as plt


def validate(model, val_loader, device, epoch, min_dist=5, show_plot=False):
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    criterion = nn.CrossEntropyLoss()
    model.eval()
    progress_bar = tqdm(total=len(val_loader), desc="Validation", unit="batch")
    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            image: torch.Tensor = batch[0].float().to(device)
            out = model(image)
            gt = batch[1].clone().detach().long().to(device)
            # gt = torch.tensor(gt, dtype=torch.long, device=device)
            loss = criterion(out, gt)
            losses.append(loss.item())
            # metrics
            output = out.argmax(dim=1).detach().cpu().numpy()
            for i in range(len(output)):
                x_pred, y_pred = postprocess(output[i])
                x_gt = batch[2][i]
                y_gt = batch[3][i]
                vis = batch[4][i]
                if show_plot:
                    out = output[i]
                    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5))
                    mask = gt.cpu().reshape((360, 640))
                    image = image.cpu()[0, :3, :, :].transpose(0, 2).transpose(0, 1)
                    ax0.imshow(image)
                    ax1.imshow(mask, cmap="viridis")
                    ax2.imshow(out.reshape((360, 640)), cmap="viridis")
                    plt.title(f"Visu: {vis}")
                    plt.show()
                if x_pred:
                    if vis != 0:
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < min_dist:
                            tp[vis] += 1
                        else:
                            fp[vis] += 1
                    else:
                        fp[vis] += 1
                if not x_pred:
                    if vis != 0:
                        fn[vis] += 1
                    else:
                        tn[vis] += 1

            progress_bar.set_postfix(
                {
                    "loss": round(np.mean(losses), 6),
                    "tp": sum(tp),
                    "tn": sum(tn),
                    "fp": sum(fp),
                    "fn": sum(fn),
                },
                refresh=True,
            )
            progress_bar.update()
            # print(
            #     "val | epoch = {}, iter = [{}|{}], loss = {}, tp = {}, tn = {}, fp = {}, fn = {} ".format(
            #         epoch,
            #         iter_id,
            #         len(val_loader),
            #         round(np.mean(losses), 6),
            #         sum(tp),
            #         sum(tn),
            #         sum(fp),
            #         sum(fn),
            #     )
            # )
        # break
    progress_bar.close()
    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    vc1 = tp[1] + fp[1] + tn[1] + fn[1]
    vc2 = tp[2] + fp[2] + tn[2] + fn[2]
    vc3 = tp[3] + fp[3] + tn[3] + fn[3]
    recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    print(
        "precision = {}".format(precision)
        + " | recall = {}".format(recall)
        + " | f1 = {}".format(f1)
    )

    return np.mean(losses), precision, recall, f1


def postprocess(feature_map, scale=2):
    feature_map *= 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7,
    )
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0] * scale
            y = circles[0][0][1] * scale
    return x, y
