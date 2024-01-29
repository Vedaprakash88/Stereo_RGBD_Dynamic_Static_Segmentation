import numpy as np
from scipy.interpolate import griddata
import cv2

def densify_depth_map(sparse_depth_map, dense_shape):
    # Create a grid of coordinates for the dense depth map
    dense_x, dense_y = np.meshgrid(np.arange(0, dense_shape[1]), np.arange(0, dense_shape[0]))

    # Find non-zero (valid) depth values and their corresponding coordinates
    valid_mask = sparse_depth_map > 0
    valid_coords = np.column_stack((np.nonzero(valid_mask)[1], np.nonzero(valid_mask)[0]))

    # Extract valid depth values
    valid_depth_values = sparse_depth_map[valid_mask]

    # Perform Inverse Distance Weighting (IDW) interpolation
    dense_depth_map = griddata(valid_coords, valid_depth_values, (dense_x, dense_y), method='cubic')

    return dense_depth_map

# Example usage:
# Replace 'sparse_depth_map.npy' and 'dense_shape' with your sparse depth map and desired dense shape
sparse_depth_map = np.load('sparse_depth_map.npy')
dense_shape = (480, 640)  # Replace with the desired dense depth map shape

dense_depth_map = densify_depth_map(sparse_depth_map, dense_shape)

# Save or visualize the dense depth map as needed
cv2.imshow('Dense Depth Map', (dense_depth_map / dense_depth_map.max() * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
