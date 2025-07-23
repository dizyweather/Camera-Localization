import numpy as np

# input parameters you want to test IN METERS
distances_apart = [0.25, 0.5, 0.75] # From each lens
distances_away = [0.25, 0.5, 0.75, 1] # From the center line

# Camera parameters IN DEGREES
horizontal_fov = 68  
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
# print("\nHorizontal Overlap (m):")
# print(df)

# Add first row of resolution data to the dataframe
df_resolution = pd.DataFrame(results[:, :, 1], index=distances_apart, columns=distances_away)
df_resolution.columns.name = 'Distance Away (m)'
df_resolution.index.name = 'Distance Apart (m)'

snip = df_resolution.iloc[0,0:].to_frame().T
snip = snip.rename(index={snip.index[0]: 'resolution (pixels/cm)'})

# print(snip)
result = pd.concat([df, snip])

result.columns.name = 'Distance Away (m)'
result.index.name = 'Distance Apart (m)'
# result = result.rename(index={result.index[4]: 'resolution (pixels/cm)'})
print(result) 

# # Save the result to a CSV file
result.to_csv('fov_results.csv', index=True)





