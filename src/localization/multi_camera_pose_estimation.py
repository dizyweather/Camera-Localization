import cv2 as cv
import cv2.aruco as aruco
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import json



##### USER DEFINED PARAMTERS
CAM1_JSON = "Y:\\Swarm Assembly 2025\\S02\\0722\\calibration_data\\small_calibration_board\\intrinsics\\small_board_left_gate_intrinsics.json" # path to the camera 1 json file
CAM2_JSON = "Y:\\Swarm Assembly 2025\\S02\\0722\\calibration_data\\small_calibration_board\\intrinsics\\small_board_right_gate_intrinsics.json" # path to the camera 2 json file, can be NONE if not used

CAM1_FRAMES_PATH = "Y:\\Swarm Assembly 2025\\S02\\0722\\calibration_data\\small_calibration_board\\frames\\gate_left" # path to the camera 1 frames
CAM2_FRAMES_PATH = "Y:\\Swarm Assembly 2025\\S02\\0722\\calibration_data\\small_calibration_board\\frames\\gate_right" # path to the camera 2 frames, can be NONE

MARKER_LENGTH_METERS = 0.049 # length of april tag marker in meters

#####

# Rip out data from json files
with open(CAM1_JSON) as f:  
    cam1 = json.load(f)

cam1_dist_coeffs = np.array(cam1['distCoeffs'])
cam1_matrix = np.array(cam1['K'])

if CAM2_JSON is not None and CAM2_JSON != "":  
    with open(CAM2_JSON) as f:  
        cam2 = json.load(f)

    cam2_dist_coeffs = np.array(cam2['distCoeffs'])
    cam2_matrix = np.array(cam2['K'])

# defines the aruco marker's object points in the marker's frame
obj_points = np.array([
    [-MARKER_LENGTH_METERS / 2.0,  MARKER_LENGTH_METERS / 2.0, 0.0],  # top-left
    [ MARKER_LENGTH_METERS / 2.0,  MARKER_LENGTH_METERS / 2.0, 0.0],  # top-right
    [ MARKER_LENGTH_METERS / 2.0, -MARKER_LENGTH_METERS / 2.0, 0.0],  # bottom-right
    [-MARKER_LENGTH_METERS / 2.0, -MARKER_LENGTH_METERS / 2.0, 0.0],  # bottom-left
], dtype=np.float32).reshape(-1, 1, 3)

# Get all the frames from camera 1 and camera 2 and pair them
cam1_frames = glob.glob(CAM1_FRAMES_PATH + "/*.JPG")  # Get all JPG files in the cam1 frames path
cam2_frames = glob.glob(CAM2_FRAMES_PATH + "/*.JPG") if bool(CAM2_FRAMES_PATH) and bool(CAM2_JSON) else None

images = zip(cam1_frames, cam2_frames) if cam2_frames is not None else zip(cam1_frames)

# set up aruco code detector params
dict = cv.aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

detector_params = cv.aruco.DetectorParameters()
detector_params.markerBorderBits = 1

detector = cv.aruco.ArucoDetector(dict, detectorParams=detector_params)

# Iterate through each image pair
index  = 0
for image_path in images:
    
    # Read the image
    cam1_image_path = image_path[0]
    cam2_image_path = image_path[1] if len(image_path) > 1 else None

    cam1_image = cv.imread(cam1_image_path)
    cam2_image = cv.imread(cam2_image_path) if cam2_image_path else None

    # Check if valid images were found
    if cam1_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    if cam2_image is None and CAM2_JSON is not None:
        raise FileNotFoundError(f"Image not found at {cam2_image_path}")
    
    # Detect markers in images
    cam1_results = detector.detectMarkers(cam1_image)
    cam2_results = detector.detectMarkers(cam2_image) if cam2_image is not None else None

    cam1_num_markers = len(cam1_results[0])
    cam1_rvecs = np.zeros((cam1_num_markers, 3), dtype=np.float32)
    cam1_tvecs = np.zeros((cam1_num_markers, 3), dtype=np.float32)

    if cam2_results is not None:
        cam2_num_markers = len(cam2_results[0])
        cam2_rvecs = np.zeros((cam2_num_markers, 3), dtype=np.float32)
        cam2_tvecs = np.zeros((cam2_num_markers, 3), dtype=np.float32)
    

    # If markers are detected, estimate pose
    if cam1_results[1] is not None:
        for i in range(cam1_num_markers):
            cv.solvePnP(obj_points, cam1_results[0][i], cam1_matrix, cam1_dist_coeffs, cam1_rvecs[i], cam1_tvecs[i])

        aruco.drawDetectedMarkers(cam1_image, corners=cam1_results[0], ids=cam1_results[1])

        pixels_in_axis = cam1_results[0][0][0] - cam1_results[0][0][0][2]

        for i in range(len(cam1_results[1])):
            cv.drawFrameAxes(cam1_image, cam1_matrix, cam1_dist_coeffs, cam1_rvecs[i], cam1_tvecs[i], MARKER_LENGTH_METERS * 1.5, 2)

    
    if cam2_results is not None and cam2_results[1] is not None:
        for i in range(cam2_num_markers):
            cv.solvePnP(obj_points, cam2_results[0][i], cam2_matrix, cam2_dist_coeffs, cam2_rvecs[i], cam2_tvecs[i])

        aruco.drawDetectedMarkers(cam2_image, corners=cam2_results[0], ids=cam2_results[1])

        for i in range(len(cam2_results[1])):
            cv.drawFrameAxes(cam2_image, cam2_matrix, cam2_dist_coeffs, cam2_rvecs[i], cam2_tvecs[i], MARKER_LENGTH_METERS * 1.5, 2)


    # Show the image with detected markers side by side
    if cam2_image is not None:
        combined_image = np.hstack((cam1_image, cam2_image))
        cv.namedWindow('detected aruco markers', cv.WINDOW_NORMAL)
        cv.imshow('detected aruco markers', combined_image)
    else:
        print('hit')
        cv.namedWindow('detected aruco markers', cv.WINDOW_NORMAL)
        cv.imshow('detected aruco markers', cam1_image)
    
    cv.waitKey(0)
    
    # Convert rvec and tvec to transformation matrix
    R1_cam2marker, _ = cv.Rodrigues(cam1_rvecs[i])
    t1_cam2marker = cam1_tvecs[i].reshape(3, 1)

    # Compose transformation matrix [R|t]
    T1_cam2marker = np.hstack((R1_cam2marker, t1_cam2marker))  # 3x4 matrix
    T1_cam2marker = np.vstack((T1_cam2marker, [0, 0, 0, 1]))   # 4x4 homogeneous

    # Invert to get T_marker2cam
    T1_marker2cam = np.linalg.inv(T1_cam2marker)

    # Extract camera position in marker frame
    camera1_position_in_marker = T1_marker2cam[:3, 3]
    # print(f"Camera position relative to marker in {image_path}: {camera_position_in_marker}")

    # Convert rotation matrix to Euler angles (XYZ convention)
    cam1_euler_angles = R.from_matrix(T1_marker2cam[:3, :3]).as_euler('xyz', degrees=True)
    # print("Camera orientation (XYZ Euler angles in degrees):", euler_angles)

    # if cam2_image is not None:
    #     R2_cam2marker, _ = cv.Rodrigues(cam2_rvecs[i])
    #     t2_cam2marker = cam2_tvecs[i].reshape(3, 1)

    #     T2_cam2marker = np.hstack((R2_cam2marker, t2_cam2marker))
    #     T2_cam2marker = np.vstack((T2_cam2marker, [0, 0, 0, 1]))

    #     T2_marker2cam = np.linalg.inv(T2_cam2marker)

    #     # Extract camera position in marker frame
    #     camera2_position_in_marker = T2_marker2cam[:3, 3]
    #     # print(f"Camera position relative to marker in {image_path}: {camera_position_in_marker}")

    #     # Convert rotation matrix to Euler angles (XYZ convention)
    #     cam2_euler_angles = R.from_matrix(T2_marker2cam[:3, :3]).as_euler('xyz', degrees=True)

    #     print(np.linalg.norm(camera1_position_in_marker - camera2_position_in_marker))
    # else:
    #     print(camera1_position_in_marker)
    print(camera1_position_in_marker)
    
    # # Save the transformation matrix & rotation using pickle
    # with open('dist_results.txt', 'a+') as f:
    #     f.write(f'[{index}] ')
    #     f.write('Distance from camera: ' + str(camera_position_in_marker[2]) + '\n')
    #     f.write('Pixels per cm at above distance away: ' + str(pixels_in_axis[0] / 5) + '\n\n')

    
    index += 1
