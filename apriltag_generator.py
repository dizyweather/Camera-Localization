import cv2
import cv2.aruco as aruco
import numpy as np

# Define grid size
cols, rows = 4, 3
tag_size = 180  # in pixels
margin = 40     # in pixels

# Load AprilTag dictionary
dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

# Create a white canvas
canvas_height = rows * tag_size + (rows + 1) * margin
canvas_width = cols * tag_size + (cols + 1) * margin
board = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

# Place AprilTags
tag_id = 0
for r in range(rows - 1, -1, -1):
    for c in range(cols):
        tag = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size)

        # rotate tag 180 degrees
        tag = cv2.rotate(tag, cv2.ROTATE_180)
        
        y = margin + r * (tag_size + margin)
        x = margin + c * (tag_size + margin)
        board[y:y+tag_size, x:x+tag_size] = tag
        tag_id += 1

for r in range(rows + 1):
    for c in range(cols + 1):
        # generate a matlike of a black square equal to margin size
        y = r * (tag_size + margin)
        x = c * (tag_size + margin)
        board[y:y+margin, x:x+margin] = 0

# Save to file
cv2.imwrite("apriltag_board.png", board)

# import cv2
# print("cv2 version:", cv2.__version__)

# import cv2.aruco as aruco
# print("Aruco module loaded")

# # Try drawing a marker
# tag = aruco.generateImageMarker(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), 0, 200)
# print("Tag shape:", tag.shape)
