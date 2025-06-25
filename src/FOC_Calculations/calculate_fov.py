import numpy as np

# input parameters you want to test IN METERS
distances_apart = [0.5, 1] # From each lens
distances_away = [0.5, 1] # From the center line

# Camera parameters IN DEGREES
horizontal_fov = 68  
# vertical_fov = 60    

horizontal_resolution = 1440  # pixels
# vertical_resolution = 1080  # pixels

results = np.zeros((len(distances_apart), len(distances_away), 2))  # To store the results

for i in range(len(distances_apart)):
    for j in range(len(distances_away)):
        # Calculate the horizontal and vertical FOV in radians
        half_horizontal_fov_rad = np.deg2rad(horizontal_fov) / 2
        # half_vertical_fov_rad = np.deg2rad(vertical_fov) / 2

        current_distance_apart = distances_apart[i]
        current_distance_away = distances_away[j]

        horizontal_overlap = 2 * np.tan(half_horizontal_fov_rad) * current_distance_away - current_distance_apart
        
        resolution = horizontal_resolution / (current_distance_away * np.tan(half_horizontal_fov_rad) * 2 * 100)

        results[i, j, 0] = horizontal_overlap
        results[i, j, 1] = resolution

# Print the results
for i in range(len(distances_apart)):
    for j in range(len(distances_away)):
        print(f"Distance Apart: {distances_apart[i]} m, Distance Away: {distances_away[j]} m -> "
              f"Horizontal Overlap: {results[i, j, 0]:.2f} m, Resolution: {results[i, j, 1]:.2f} pixels/cm")
        





