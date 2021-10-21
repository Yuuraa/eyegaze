import cv2
import argparse
import numpy as np
from tqdm import tqdm
import glob
import cv2
import json
#%%
DATA_PATH = "/mnt/disk1/CNUH/CNUH/videos"
SAVE_PATH = "/mnt/disk1/CNUH/resized_vids"
META_PATH = "/mnt/disk1/CNUH/metadata"
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

parser = argparse.ArgumentParser()
parser.add_argument("--video_num", type=int, default=1)
args = parser.parse_args()

video_path = glob.glob(f"{DATA_PATH}/{str(args.video_num).zfill(3)}.*")[0]

print(f"Reading {video_path}")

cap = cv2.VideoCapture(video_path)
video_name = video_path.split("/")[-1].split(".")[0]
ext = video_path.split("/")[-1].split(".")[-1]
metadata = json.load(open(f"{META_PATH}/{video_name}.json", "r"))
w, h, fps = metadata['w'], metadata['h'], metadata['fps']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_AREA)
    cv2.imshow("Resized", frame)
    if cv2.waitKey(30) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()