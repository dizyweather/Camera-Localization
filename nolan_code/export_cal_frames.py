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
    # USER DEFINED PARAMTERS
    # - Can provide 1 path or 2 paths (with dk)
    # - if path 2 None, export frames for path 1 only

    CAM1_VIDEO  = "./data/videos/GH010185.MP4"  # MP4 path 1
    CAM2_VIDEO  = None   # MP4 path 2 | "" | None
    DELTA       = 0                    # cam1_frame = cam2_frame + DELTA

    CAM1_FRAMES = [
        frame for frame in range((0 * 60 + 0) * 60, (0 * 60 + 45) * 60 , 51)    # (Start Frame, End Frame, Number of frames to take)
    ]
    OUTPUT_ROOT = "./data/calibration_frames"  # Output root folder

    # Output folders:
    # <OUTPUT_ROOT>/cam1_cal/
    # <OUTPUT_ROOT>/cam2_cal/   (created only when video 2 is valid)

    # END EDIT SECTION

    cam1_dir = os.path.join(OUTPUT_ROOT, "cam1_cal")
    ensure_dir(cam1_dir)

    print("Extracting cam1 frames")
    extract_frames(CAM1_VIDEO, CAM1_FRAMES, cam1_dir, "cam1")

    has_cam2 = bool(CAM2_VIDEO) and os.path.isfile(CAM2_VIDEO)
    if has_cam2:
        CAM2_FRAMES = [f - DELTA for f in CAM1_FRAMES]
        cam2_dir = os.path.join(OUTPUT_ROOT, "cam2_cal")
        ensure_dir(cam2_dir)

        print("\nExtracting cam2 frames")
        extract_frames(CAM2_VIDEO, CAM2_FRAMES, cam2_dir, "cam2")
    else:
        print("\nSecond video not provided or unreadable - skipping cam2 extraction")

    print("\nDone")


if __name__ == "__main__":
    main()