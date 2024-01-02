import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
	FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
	RasterizationSettings, MeshRenderer, MeshRasterizer,
	BlendParams,
	SoftSilhouetteShader, HardPhongShader, PointLights,
	TexturesVertex
)

if torch.cuda.is_available():
	device = torch.device("cuda:0")
else:
	device = torch.device("cpu")
	print(f"WARNING: CPU only, this will be slow!")
 
# now we create output directory for rendered images from each optimization iteration
# so we can see the process step-by-step
output_dir = "./result_teapot_si"
 
# we load mesh model
# since the model doesn't come with textures (material colors), we make an
# all one tensor and set that as the texture tensor for the model
verts, faces_idx, _ = load_obj("./data/teapot.obj")
faces = faces_idx.verts_idx

verts_rgb = torch.ones_like(verts)[None] # (1,V,3) 
textures = TexturesVertex(verts_features=verts_rgb.to(device))

teapot_mesh = Meshes(verts=[verts.to(device)],
                     faces=[faces.to(device)],
                     textures=textures)

# define the cmaera model
cameras = FoVPerspectiveCameras(device=device)


# the next block of code defines a differntiable renderer
# we need a rasterizer and a shader and a few hyperparameters
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1. / 1e-4 - 1.)*blend_params.sigma,
    faces_per_pixel=100
)

silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1
)

lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device,
                           cameras=cameras,
                           lights=lights)
)

# next we define the camera location and compute corresponding rotation, R and
# displacement, T of the camera. This rotation and displacement are the target
# camera position, that is, we are going to generate an image from this camera position and
# use the image as the observed image in our problem
distance = 3
elevation = 50.0
azimuth = 0.0

R, T, camera_center = look_at_view_transform(distance, elevation, azimuth, device=device)
print(f"Camera center: {camera_center}")

# no we can generate an image, image_ref from this camera position
cameras[0].