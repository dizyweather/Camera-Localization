# Nolan's code edited to be used for bee swarm experiment
from pathlib import Path
import cv2
import numpy as np
import json
from tqdm import tqdm
from pupil_apriltags import Detector
import os

########### EDIT BELOW HERE
CALIBRATION_BOARD_TYPE = "large"  # "small" or "large"
CAMERA_ID = "shed_left"  # "gate_left" or "gate_right", "shed_left" or "shed_right"
IMAGE_FOLDER_PATH = "Y:\\Swarm Assembly 2025\\S02\\0725"  # image folder path

EXT               = "JPG"          # PNG, JPG, case insensitive

N_THREADS         = 20           # For multi-processing
MAX_PROCESSING_WIDTH = 1920 
SHOW_DETECTIONS   = True

############# DON'T EDIT BELOW

IMG_DIR           = Path(f"{IMAGE_FOLDER_PATH}\\calibration_data\\{CALIBRATION_BOARD_TYPE}_calibration_board\\frames\\{CAMERA_ID}")  # image folder path 
SAVE_JSON         = True           # Export intrinsics matrix K
OUT_FILE          = Path(f"{IMAGE_FOLDER_PATH}\\calibration_data\\{CALIBRATION_BOARD_TYPE}_calibration_board\\intrinsics\\{CALIBRATION_BOARD_TYPE}_board_{CAMERA_ID}_intrinsics.json")    # Output file name

# Nolan's calibration board (large): tagsize = 75mm, tag space = 40mm, 4x5 grid
# Daniel's calibration board (small): tagsize = 49mm, tag space = 11mm, 3x4 grid

if CALIBRATION_BOARD_TYPE == "large":
    TAG_SIZE_MM       = 75             # tag edge (mm)
    TAG_SPACING_MM    = 40              # tag gap (mm)
    GRID_ROWS, GRID_COLS = 4, 5
    TAG_ID_OFFSET     = 0
elif CALIBRATION_BOARD_TYPE == "small":
    TAG_SIZE_MM       = 49             # tag edge (mm)
    TAG_SPACING_MM    = 11              # tag gap (mm)
    GRID_ROWS, GRID_COLS = 3, 4
    TAG_ID_OFFSET     = 0
else:
    raise ValueError("CALIBRATION_BOARD_TYPE must be 'large' or 'small'")

# Detector config
at = Detector(
    families="tag36h11",
    nthreads=N_THREADS,
    quad_decimate=1.0,
    quad_sigma=0.0,
)

# Helper function
def to_uint8(gray):
    if gray.dtype == np.uint8:
        return gray
    g_min, g_max = int(gray.min()), int(gray.max())
    scale = 255.0 / max(1, g_max - g_min)
    return cv2.convertScaleAbs(gray, alpha=scale, beta=-g_min * scale)

def grid_object_pts(tag_id):
    idx = tag_id - TAG_ID_OFFSET
    r, c = divmod(idx, GRID_COLS)
    base_x = c * (TAG_SIZE_MM + TAG_SPACING_MM)
    base_y = r * (TAG_SIZE_MM + TAG_SPACING_MM)
    return np.array([
        [base_x,               base_y,               0.0],
        [base_x + TAG_SIZE_MM, base_y,               0.0],
        [base_x + TAG_SIZE_MM, base_y + TAG_SIZE_MM, 0.0],
        [base_x,               base_y + TAG_SIZE_MM, 0.0],
    ], dtype=np.float32)

def detect_tags(bgr):
    gray = to_uint8(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    h, w = gray.shape
    scale_back = 1.0

    if w > MAX_PROCESSING_WIDTH:
        f = MAX_PROCESSING_WIDTH / w
        gray = cv2.resize(gray, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_AREA)
        scale_back = 1 / f

    dets = at.detect(gray)
    for d in dets:
        d.corners *= scale_back
    return dets

# Start Main Area
objpoints, imgpoints = [], []
ext_glob = f"*.{EXT.lower()}"
files = sorted(list(IMG_DIR.glob(ext_glob)) + list(IMG_DIR.glob(ext_glob.upper())))
print(f"Scanning {len(files)} files")

for path in tqdm(files):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        continue
    detections = detect_tags(img)
    if len(detections) < 1:
        print(f"No tags found in {path.name}")
        continue

    frame_obj, frame_img = [], []
    for det in detections:
        tid = det.tag_id
        if not TAG_ID_OFFSET <= tid < TAG_ID_OFFSET + GRID_ROWS * GRID_COLS:
            continue
        frame_obj.extend(grid_object_pts(tid))
        frame_img.extend(det.corners.astype(np.float32))

        if SHOW_DETECTIONS:
            corners_for_drawing = det.corners.astype(int)
            for p in corners_for_drawing:
                cv2.circle(img, tuple(p), 4, (0, 255, 0), 2)

    if len(frame_obj) >= 4:
        objpoints.append(np.asarray(frame_obj, np.float32))
        imgpoints.append(np.asarray(frame_img, np.float32))

    if SHOW_DETECTIONS:
        display_h, display_w = img.shape[:2]
        if display_w > 1920:
            scale = 1920 / display_w
            img_display = cv2.resize(img, (0,0), fx=scale, fy=scale)
        else:
            img_display = img
        cv2.imshow("dbg", img_display)
        if cv2.waitKey(1) == 27:   #Esc key
            SHOW_DETECTIONS = False
            cv2.destroyAllWindows()
cv2.destroyAllWindows()

if len(objpoints) < 3:
    raise RuntimeError("Need >= 3 valid images; found %d" % len(objpoints))
print(f"Using {len(objpoints)} images for calibration.")

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
flags = None

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img.shape[1::-1],
    None, # cameraMatrix initial guess (can be None)
    None, # distCoeffs initial guess (can be None)
    flags=0, # No special distortion model flags, use default polynomial
    criteria=criteria
)

print("\nRESULTS:")
print("RMS reprojection error:", rms)
print("\nK:\n", K)

# dist_poly will contain k1, k2, p1, p2, k3, (k4, k5, k6 optional)
print(f"\nDist coeffs array shape: {dist.shape}")
print("Dist coeffs (k1,k2,p1,p2,k3,...):\n", dist.ravel())

if SAVE_JSON:
    #folder name
    os.makedirs(OUT_FILE.parent, exist_ok=True) ## TEST LATER
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump({
            "K": K.tolist(),
            "distCoeffs": dist.ravel().tolist(),
            "rms": float(rms),
            "flags": flags,
            "image_size": img.shape[1::-1],
        }, fh, indent=2)
    print("Saved", OUT_FILE)