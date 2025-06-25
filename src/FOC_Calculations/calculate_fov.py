import numpy as np

# input parameters you want to test IN METERS
distances_apart = [0.5, 1, 1.5, 2] # From each lens
distances_away = [0.5, 1, 1.5, 2] # From the center line

# Camera parameters IN DEGREES
horizontal_fov = 122.6	  
# vertical_fov = 60    

horizontal_resolution = 1440  # pixels
# vertical_resolution = 1080  # pixels

# results[i] = The ith distance apart
# results[j] = The jth distance away
# results[i, j, 0] = Horizontal overlap in meters
# results[i, j, 1] = Resolution in pixels/cm
results = np.zeros((len(distances_apart), len(distances_away), 2))  

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

# pandas display
import pandas as pd
df = pd.DataFrame(results[:, :, 0], index=distances_apart, columns=distances_away)
df.columns.name = 'Distance Away (m)'
df.index.name = 'Distance Apart (m)'
print("\nHorizontal Overlap (m):")
print(df)

df_resolution = pd.DataFrame(results[:, :, 1], index=distances_apart, columns=distances_away)
df_resolution.columns.name = 'Distance Away (m)'
df_resolution.index.name = 'Distance Apart (m)'
print("\nResolution (pixels/cm):")
print(df_resolution)
        
# combine the two dataframes
combined_df = pd.concat([df, df_resolution.add_suffix(' Resolution (pixels/cm)')], axis=1)

combined_df.to_csv('fov_results.csv', index=True)




