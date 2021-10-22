from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import os

# Arguments
SEG_PATH = "/mnt/disk1/CNUH/resize_results"
SAVE_PATH = "/mnt/disk1/CNUH/resize_track_results"
FPS = 30
WIDTH, HEIGHT = 640, 480
os.makedirs(SAVE_PATH, exist_ok=True)
MAX_BUFFER_SIZE = 64 # Max length of buffer for previous center points

# =================== !!!!!!!!!!!!!!!!! FIXME !!!!!!!!!!!!!!!!! Edit path!! ===================
vs = cv2.VideoCapture(f"{SEG_PATH}/003.mp4")
out = cv2.VideoWriter(
        os.path.join(SAVE_PATH, '003.mp4'),
        cv2.VideoWriter_fourcc(*'XVID'),
        FPS,
        (WIDTH, HEIGHT)
)
# =======================================================================================

# For tracking previous center of pupils
pt1s = deque(maxlen=MAX_BUFFER_SIZE)
pt2s = deque(maxlen=MAX_BUFFER_SIZE)


# keep looping
while True:
    ret, frame = vs.read()
    if not ret:
	    break


    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # mask = cv2.inRange(frame, (240, 240, 240), (300, 300, 300))
    mask = cv2.inRange(frame, (220, 220, 220), (300, 300, 300))
    cv2.imshow("Mask", mask)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2) 

    mask1 = mask[:, :WIDTH//2]
    mask2 = mask[:, WIDTH//2:]

    # """
    cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts1 = imutils.grab_contours(cnts1)
    cnts2 = imutils.grab_contours(cnts2)
    center1, center2 = None, None
    # only proceed if at least one contour was found

    if len(cnts1) > 0:
    	# find the largest contour in the mask, then use it to compute the minimum enclosing circle & centroid
    	c = max(cnts1, key=cv2.contourArea)
    	((x, y), radius) = cv2.minEnclosingCircle(c)
    	M = cv2.moments(c)
    	center1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    	if radius > 10:
    		cv2.circle(frame, (int(x), int(y)), int(radius),
    			(0, 255, 255), 2)
    		cv2.circle(frame, center1, 5, (0, 0, 255), -1)
    # update the points queue
    pt1s.appendleft(center1)

    if len(cnts2) > 0:
    	# find the largest contour in the mask, then use it to compute the minimum enclosing circle & centroid
    	c = max(cnts2, key=cv2.contourArea)
    	((x, y), radius) = cv2.minEnclosingCircle(c)
    	M = cv2.moments(c)
    	center2 = (int(M["m10"] / M["m00"]) + WIDTH//2, int(M["m01"] / M["m00"]))
    	if radius > 10:
    		cv2.circle(frame, (int(x) + WIDTH//2, int(y)), int(radius),
    			(0, 255, 255), 2)
    		cv2.circle(frame, center2, 5, (0, 0, 255), -1)
    # update the points queue
    pt2s.appendleft(center2)

    # loop over the set of tracked points
    for i in range(1, len(pt1s)):
    	if pt1s[i - 1] is None or pt1s[i] is None:
    		continue
    	# otherwise, compute the thickness of the line and
    	# draw the connecting lines
    	thickness = int(np.sqrt(MAX_BUFFER_SIZE / float(i + 1)) * 2.5)
    	cv2.line(frame, pt1s[i - 1], pt1s[i], (0, 0, 255), thickness)

    for i in range(1, len(pt2s)):
    	if pt2s[i - 1] is None or pt2s[i] is None:
    		continue
    	# otherwise, compute the thickness of the line and
    	# draw the connecting lines
    	thickness = int(np.sqrt(MAX_BUFFER_SIZE / float(i + 1)) * 2.5)
    	cv2.line(frame, pt2s[i - 1], pt2s[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
        vs.release()
        out.release()
# close all windows
out.release()
cv2.destroyAllWindows()