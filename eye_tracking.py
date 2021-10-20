import cv2
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--video_num", type=int, default=1)
args = parser.parse_args()

DATA_PATH = "/mnt/disk1/CNUH/CNUH/videos"
VIDEO_PATH = glob.glob(f"{DATA_PATH}/{str(args.video_num).zfill(3)}.*")[0]

cap = cv2.VideoCapture(VIDEO_PATH)

def print_centerpoint(roi, cX, cY):
    cv2.putText(roi, f"({cX},{cY})", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

def filter_draw_contour(roi, contours, prev_center, r, h):
    if not contours:
        return None, prev_center
    def get_center(contour):
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print_centerpoint(roi, cX, cY)
        return (cX, cY)

    centers = [get_center(contour) for contour in contours]
    center_dists = [(prev_center[0] - center[0])**2 + (prev_center[1] - center[1])**2 for center in centers]
    if min(center_dists) > r**2//16:
        return None, prev_center
    min_idx = min([i for i in range(len(centers))], key=lambda i: center_dists[i])

    cv2.drawContours(roi, contours, min_idx, (242, 255, 89), cv2.FILLED)

    return contours[min_idx], centers[min_idx]


i = 0
while True:
    # Read video frame
    ret, frame = cap.read()
    r, c, _ = frame.shape
    cv2.imshow('Frame', frame)
    if i == 0: 
        prev_center1 = [r//4, c//4]
        prev_center2 = [r//4, c//4]

    # ROI: eye region
    roi1 = frame[0:r//2, :c//2]
    roi2 = frame[0:r//2, c//2:]

    # print(r, c)
    thres_1 = max(40, np.mean(roi1) - 70)
    thres_2 = max(40, np.mean(roi2) - 70)

    roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    _, thres1 = cv2.threshold(roi1, thres_1, 255, cv2.THRESH_BINARY_INV)
    _, thres2 = cv2.threshold(roi2, thres_2, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5, 5), np.uint8)
    thres1 = cv2.morphologyEx(thres1, cv2.MORPH_OPEN, kernel)
    thres2 = cv2.morphologyEx(thres2, cv2.MORPH_OPEN, kernel)
    erosion_size = (10, 10)
    dilate_size = (8, 8)
    thres1 = cv2.erode(thres1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erosion_size))
    thres1 = cv2.dilate(thres1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_size))
    contours1, _ = cv2.findContours(thres1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thres2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour1, prev_center1 = filter_draw_contour(roi1, contours1, prev_center1, r, c)
    contour2, prev_center2 = filter_draw_contour(roi2, contours2, prev_center2, r, c)
    
    # cv2.drawContours(roi2, contours2, -1, (255, 255, 255), cv2.FILLED)
    thres = cv2.hconcat([thres1, thres2])
    cv2.imshow('threshold', thres)

    cv2.imshow('roi1', roi1)
    cv2.imshow('roi2', roi2)
    if cv2.waitKey(30) == 27:
        break
    i += 1

cv2.destroyAllWindows()