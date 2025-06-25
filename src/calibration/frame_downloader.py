# Downloads frames from a video file for calibration purposes.
# Should be a video of moving the checkboard around in the camera's view.
# Saves the frames in a specified directory for calculate_calibration_values.py to use.

import cv2 as cv
import numpy as np
import os

# Absolute path to the current file
file_path = os.path.dirname(__file__)

# Where to store the training frames
download_path = file_path + '/../../data/calibration_frames'
try:
    os.mkdir(download_path)

except FileExistsError:
    print(f"Directory {download_path} already exists. Continuing to download frames.")

# path to video file
video_path = file_path + '/../../data/videos/aruco_dist.MP4'
cap = cv.VideoCapture(video_path)

start_frame = 0
end_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - 1
number_of_wanted_frames = 25  # you want to download 10 evenly spaced frames from the interval [start_frame, end_frame]

# how many frames to download
frames = np.linspace(start_frame, end_frame, number_of_wanted_frames, dtype=int)

for frame in frames:
    cap.set(cv.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    if ret:
        cv.imwrite(f'{download_path}/frame_{frame}.png', img)
        print(f"Downloaded frame {frame}")
    else:
        print(f"Failed to retrieve frame {frame}")