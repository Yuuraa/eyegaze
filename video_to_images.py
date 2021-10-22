import cv2
import numpy as np
import argparse
import glob
import os
import tqdm

DATA_PATH = "/mnt/disk1/CNUH/CNUH/videos"
SAVE_PATH = "/mnt/disk1/CNUH/CNUH/video_images/test/images"


def save_vid_to_img(i):
    video_name = str(i).zfill(3)
    video_path = glob.glob(f"{DATA_PATH}/{video_name}.*")[0]

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while True:
        # Read video frame
        ret, frame = cap.read()
        if not ret: 
            return
        cv2.imwrite(f'{SAVE_PATH}/{video_name}_{frame_id}.png', frame)
        frame_id += 1


if __name__ == "__main__":
    for fname in tqdm.tqdm(os.listdir(DATA_PATH)):
        i = fname.split(".")[0]
        save_vid_to_img(i)

