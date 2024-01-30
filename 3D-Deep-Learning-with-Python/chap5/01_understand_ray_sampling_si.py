import torch
import math
import numpy as np

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    look_at_view_transform,
    NDCGridRaysampler,
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    

# we define a batch of 10 cameras
# All 10 cameras point at an object located at the center of the world coordinates
num_views: int = 10
azimuth_range: float = 180
elev = torch.linspace(0,0, num_views)
azim = torch.linspace(-azimuth_range, azimuth_range, num_views) + 180.0

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# R and T transform points to align with the camera
R,T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device,R=R, T=T)


# Create our ray sampler
# for this sample we have 50 points per ray
image_size = 64
volume_extent_world = 3.0
raysampler = NDCGridRaysampler(
    image_width=image_size,
    image_height=image_size,
    n_pts_per_ray=50,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

# the sampler needs to know where our cameras are and in what directions they are pointing
ray_bundle = raysampler(cameras)

# ray_bundle contains a collection of tensors that specify the sampled rays and points
print(f"ray_bundle origins tensor shape: {ray_bundle.origins.shape}") # (10,64,64,3) - the 3 is for three numbers to specify its 3D location
print(f"ray_bundle directions tensor shape: {ray_bundle.directions.shape}") # (10,64,64,3) - the 3 is for three numbers to specify a direction in 3D spaces

print(f"ray_bundle lengths = {ray_bundle.lengths.shape}") # (10,64,64,50) - the 50 is for the 50 points on each ray

# xys is a tensor about the x and y locations on the image plane corresponding to each ray
print(f"ray_bundle xys shape = {ray_bundle.xys.shape}") # (10,64,64,2) - the 2 includes: one number for the location and one number to represent the y location

# Finally we can save our ray_bundle to a .pt file that we can use later
torch.save({
    'ray_bundle': ray_bundle,
    }, 'ray_sampling.pt')