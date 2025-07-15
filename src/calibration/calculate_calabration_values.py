# Using the frames in data/calibration_frame as training, this script calculates the camera calibration values
# and saves them in a pkl file for later use in undistort_image.py & pose_estimation.py

import numpy as np
import cv2 as cv
import glob
import pickle
import os

# Absolute path to the current file
file_path = os.path.dirname(__file__)

# checkerboard size (MAKE SURE THIS MATCHES YOUR CHECKERBOARD)
CHECKERBOARD_SIZE = (4, 5)

# get all images from the directory where you stored the training frames
images = glob.glob(file_path + '/../../data/calibration_frames/*.png')

if images == []:
    print("No images found in the specified directory. Please check the path.")
    exit()

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD_SIZE[0],0:CHECKERBOARD_SIZE[1]].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# For each image 
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        orig_img = img.copy()
        # cv.imwrite('uncalibrated.png', orig_img)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)

        print(len(objpoints), len(imgpoints))
    else:
        print(f"Chessboard corners not found in {fname}")


# Retrieve the camera matrix and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save values to a pkl file used in undistort_image.py
with open(file_path + '/../results/camera_matrix.pkl', 'wb') as f: 
    pickle.dump([mtx, dist, rvecs, tvecs], f)

