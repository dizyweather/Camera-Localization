# Nolan's code edited to be used for bee swarm experiment
import cv2
import os
from typing import List

# Helper functions 
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_total_frames(cap: cv2.VideoCapture):
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def extract_frames(
    video_path: str,
    frame_indices: List[int],
    output_dir: str,
    prefix: str,
):
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = get_total_frames(cap)
    for idx in frame_indices:
        if not (0 <= idx < total):
            print(f"Warning: {prefix} frame {idx} out of range (0–{total-1}): skipping")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"Warning: cv2 failed at {prefix} frame {idx}; skipping")
            continue

        out_path = os.path.join(output_dir, f"{prefix}_{idx:06d}.JPG")
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    cap.release()

def main():
    ########## USER DEFINED PARAMTERS
    # - Can provide 1 path or 2 paths (with dk)
    # - if path 2 None, export frames for path 1 only
    IMAGE_FOLDER_PATH       = "Y:\\Swarm Assembly 2025\\S02\\0728"  # image folder path
    STEREO_SIDE             = "shed"  # "gate" or "shed"
    CALIBRATION_BOARD_TYPE  = "large"  # "small" or "large"

    # MAKE SURE TO CHANGE VIDEO NUMBER {_______.MP4}
    CAM1_VIDEO  = f"{IMAGE_FOLDER_PATH}\\gopro_pair_{STEREO_SIDE}\\left_camera\\GH020262.MP4"  # MP4 path 1 (SHOULD BE LEFT VIDEO)
    CAM2_VIDEO  = f"{IMAGE_FOLDER_PATH}\\gopro_pair_{STEREO_SIDE}\\right_camera\\GH020199.MP4"   # MP4 path 2 (SHOULD BE RIGHT VIDEO)
    DELTA       = -9                  # cam1_frame = cam2_frame + DELTA

    CAM1_START_TIME_SECONDS = 3 * 60 + 20  # Start time of calibration in seconds for cam1
    CAM1_END_TIME_SECONDS   = 4 * 60 + 55  # End time of calibration in seconds for cam1
    USE_EVERY_NTH_FRAME = 100  # Take every Nth frame

    ########## END EDIT SECTION

    CAM1_FRAMES = [
        frame for frame in range(CAM1_START_TIME_SECONDS * 60, CAM1_END_TIME_SECONDS * 60 , USE_EVERY_NTH_FRAME)    # (Start Frame, End Frame, Take every Nth frame)
    ]
    OUTPUT_ROOT = f"{IMAGE_FOLDER_PATH}\\calibration_data\\{CALIBRATION_BOARD_TYPE}_calibration_board\\frames"  # Output root folder

    CAM1_OUTPUT_FOLDER_NAME = f"{STEREO_SIDE}_left"
    CAM2_OUTPUT_FOLDER_NAME = f"{STEREO_SIDE}_right"


    # Comment out the following lines if you know what you are doing
    if CALIBRATION_BOARD_TYPE not in ["large", "small"]:
        raise ValueError("CALIBRATION_BOARD_TYPE must be 'large' or 'small'")
    
    if STEREO_SIDE not in ["gate", "shed"]:
        raise ValueError("STEREO_SIDE must be 'gate' or 'shed'")
    
    
    cam1_dir = os.path.join(OUTPUT_ROOT, CAM1_OUTPUT_FOLDER_NAME)
    ensure_dir(cam1_dir)

    print("Extracting cam1 frames")
    extract_frames(CAM1_VIDEO, CAM1_FRAMES, cam1_dir, CAM1_OUTPUT_FOLDER_NAME + "_frame")

    has_cam2 = bool(CAM2_VIDEO) and os.path.isfile(CAM2_VIDEO)
    if has_cam2:
        CAM2_FRAMES = [f - DELTA for f in CAM1_FRAMES]
        cam2_dir = os.path.join(OUTPUT_ROOT, CAM2_OUTPUT_FOLDER_NAME)
        ensure_dir(cam2_dir)

        print("\nExtracting cam2 frames")
        extract_frames(CAM2_VIDEO, CAM2_FRAMES, cam2_dir, CAM2_OUTPUT_FOLDER_NAME + "_frame")
    else:
        print("\nSecond video not provided or unreadable - skipping cam2 extraction")

    print("\nDone")


if __name__ == "__main__":
    main()

