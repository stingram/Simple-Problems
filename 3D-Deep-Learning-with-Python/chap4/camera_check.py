from pytorch3d.renderer import look_at_view_transform
import torch

# Step 1: Obtain transformation components
dist = 2.7
elev = 210.0
azim = 180.0
R, T = look_at_view_transform(dist, elev, azim, 'cuda')

# Step 2: Combine rotation and translation into view matrix
view_matrix = torch.eye(4)
view_matrix[:3, :3] = R
view_matrix[:3, 3] = T

# Step 3: Extract camera position from view matrix
camera_position = view_matrix[:3, 3]

# Elevation and Azimuth do not effect the camera position
# WHen doing look_at_view_transform, dist is only thing that
# changes z value. z = dist
print("Camera Position:", camera_position)