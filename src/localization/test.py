import os
from pathlib import Path

OUT_FILE = "Y:\\Swarm Assembly 2025\\S02\\0725\\calibration_data\\small_calibration_board\\intrinsics\\small_board_gate_left_intrinsics.json"  # Output file name

OUT_FILE = Path(OUT_FILE)
print(OUT_FILE.parent)

with open(OUT_FILE, "w", encoding="utf-8") as fh:
    exit()