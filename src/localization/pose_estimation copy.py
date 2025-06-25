import cv2 as cv
import cv2.aruco as aruco
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

marker_length = 0.05 # length of aruco marker in meters

# Absolute path to the current file
file_path = os.path.dirname(__file__)

# Getting back the objects:
with open(file_path + '/../results/camera_matrix.pkl', 'rb') as f:  
    cam_matrix, dist_coeffs, _, _ = pickle.load(f)

# defines the aruco marker's object points in the marker's frame
obj_points = np.array([
    [-marker_length / 2.0,  marker_length / 2.0, 0.0],  # top-left
    [ marker_length / 2.0,  marker_length / 2.0, 0.0],  # top-right
    [ marker_length / 2.0, -marker_length / 2.0, 0.0],  # bottom-right
    [-marker_length / 2.0, -marker_length / 2.0, 0.0],  # bottom-left
], dtype=np.float32).reshape(-1, 1, 3)

# target image path
image_path = file_path + '/../../data/misc/frame_0.png'
image = cv.imread(image_path)

# z = 39.something
# y = 23 ish
# x = almost nothing

# 46.6
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# set up aruco code params
dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_1000)

detector_params = cv.aruco.DetectorParameters()
detector_params.markerBorderBits = 1

detector = cv.aruco.ArucoDetector(dict, detectorParams=detector_params)
results = detector.detectMarkers(image)

num_markers = len(results[0])
rvecs = np.zeros((num_markers, 3), dtype=np.float32)
tvecs = np.zeros((num_markers, 3), dtype=np.float32)

# If markers are detected, estimate pose
if results[1] is not None:
    for i in range(num_markers):
        cv.solvePnP(obj_points, results[0][i], cam_matrix, dist_coeffs, rvecs[i], tvecs[i])

    aruco.drawDetectedMarkers(image, corners=results[0], ids=results[1])

    for i in range(len(results[1])):
        cv.drawFrameAxes(image, cam_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length * 1.5, 2)

cv.namedWindow('detected aruco markers', cv.WINDOW_NORMAL)
cv.imshow('detected aruco markers', image)
while True:
    if cv.waitKey(1) == ord('q'):
        break

# Convert rvec and tvec to transformation matrix
R_cam2marker, _ = cv.Rodrigues(rvecs[i])
t_cam2marker = tvecs[i].reshape(3, 1)

# Compose transformation matrix [R|t]
T_cam2marker = np.hstack((R_cam2marker, t_cam2marker))  # 3x4 matrix
T_cam2marker = np.vstack((T_cam2marker, [0, 0, 0, 1]))   # 4x4 homogeneous

# Invert to get T_marker2cam
T_marker2cam = np.linalg.inv(T_cam2marker)

# Extract camera position in marker frame
camera_position_in_marker = T_marker2cam[:3, 3]
print(f"Camera position relative to marker {results[1][i][0]}: {camera_position_in_marker}")

# Convert rotation matrix to Euler angles (XYZ convention)
euler_angles = R.from_matrix(T_marker2cam[:3, :3]).as_euler('xyz', degrees=True)
print("Camera orientation (XYZ Euler angles in degrees):", euler_angles)

# Save the transformation matrix & rotation using pickle
with open(file_path + '/../results/pose_estimation.pkl', 'wb') as f:
    pickle.dump({
        'T_marker2cam': T_marker2cam,
        'camera_position_in_marker': camera_position_in_marker,
        'euler_angles': euler_angles
    }, f)

