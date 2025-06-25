import numpy as np
import cv2 as cv
import pickle
import os

# Absolute path to the current file
file_path = os.path.dirname(__file__)

# Path to file you want to undistort 
img = cv.imread('data/frame_0.png')

if img is None:
    print("Image not found. Please check the path.")
    exit()

# Getting back the objects:
with open(file_path + '/../results/camera_matrix.pkl', 'rb') as f:  
    mtx, dist, rvecs, tvecs = pickle.load(f)

# Select image to be undistorted
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# before
cv.imwrite(file_path + '/../results/before.png', img)

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# uncropped undistorted image
cv.imwrite(file_path + '/../results/uncropped_calibresult.png', dst)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite(file_path + '/../results/calibresult.png', dst)

cv.destroyAllWindows()