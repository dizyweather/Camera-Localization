Purpose of Code:
    Localize the camera's position from the aruco codes it can see

How to use:
In order to achieve this we first must calculate the camera's calibration values to account for distorition

1. Setup camera in desired position and capture a video of a checkerboard pattern on that camera
2. Run frame_downloader.py to collect intermidiete frames from the video to be used as training data. (can add frames manually instead)
    - Remember to change the video location inside the script
    - Will save data to data/calibration frames
3. Run calculate_calibration_values.py
    - Uses the frames in calibration_frames to calculate the distorition of the camera
    - Saves data to src/results folder
4. Run pose_estimation.py 
    - Estimates the camera's pose relative to the aruco code in the given image path

Optional:
    Run undistort_image.py to undistort a given image using the camera_matrix values calculated
    Stores results in src/results