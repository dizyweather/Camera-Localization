Purpose of Code:
    1. To be able to calibrate the gopro camera by downloading frames of calibration board and calculating intrinsics
	2. To be able to localize camera relative to the other camera in the stereo setup (IN PROGRESS)

NOTE: This code HEAVILY relies on the file structure in the entry template!!! Any change to the file structure will have to also be changed in code.
 
Calibration - How to use:
1. Download Frames
	- In "export_cal_frames.py"
	- Change the user-defined parameters
		- Remember to change the video number at the end of "CAM1_VIDEO"
		- MANUALLY find the calibration start and end times in cam1
		- Can get "DELTA" using Henry's Synchroneity MATLAB code
	- Run code
	
	This should download frames into a folder to be used by the next step

2. Calculate Intrinsics
	- In "calculate_intrinsics.py"
	- Change the user-defined parameters
	- Run code

	This should use the above frames and calculate the camera's intrinsics and saves to a json file

