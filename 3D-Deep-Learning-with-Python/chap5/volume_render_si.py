# Differentiable Volume Rendering
# This constructs 3D data from 2D images. We represent the shape and texture
# of the object as a parametric function. This function can be used to generate
# 2D projections. Given 2D projections, we can optimize the parameters of these
# implicit shape and texture functions so that its prjections are the multi-view
# 2D images. This is possible since the rendering process is completely differentiable,
# and the implicit functions are also differentiable

import os
import sys
import time
import json
import glob
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    VolumeRenderer,
    NDCGridRaysampler,
    EmissionAbsorptionRaymarcher
)
from pytorch3d.transforms import so3_exp_map

from plot_image_grid import image_grid
from generate_cow_renders import generate_cow_renders

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
# we generate 40 cameras, images, and silhouette images with different angles
target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=40)

# next, we define a ray sampler
render_size = 128
volume_extent_world = 3.0

raysampler = NDCGridRaysampler(
    image_width=render_size,
    image_height=render_size,
    n_pts_per_ray=150,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

# we create a ray marcher as before. We also define a renderer of type VolumeRenderer, which
# is a nice interface that lets ray samplers and ray marchers do all the heavy lifting under
# the hood
raymarcher = EmissionAbsorptionRaymarcher()
renderer = VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)

# Next we define VolumeModel class. It encapsulates a volume so that the gradients can be computed
# in the forward function and the volume densities and colors can be updated by the optimizer
class VolumeModel(torch.nn.Module):
    def __init__(self, renderer, volume_size=[64]*3,voxel_size=0.1):
        super().__init__()
        
        # these are values we are trying to estimate for each point in our 3D volume
        self.log_densities = torch.nn.Parameter(-4.0*torch.ones(1,*volume_size)) # (1,64,64,64)
        self.log_colors = torch.nn.Parameter(torch.zeros(3,*volume_size)) # (3,64,64,64)
        
        self._voxel_size = voxel_size
        self._renderer = renderer
        
    def forward(self, cameras):
        batch_size = cameras.R.shape[0]
        
        densities = torch.sigmoid(self.log_densities)
        colors = torch.sigmoid(self.log_colors)
        
        volumes = Volumes(
            densities=densities[None].expand(
                batch_size, *self.log_densities.shape), # (batch_size,1,64,64,64)
            features=colors[None].expand(
                batch_size, *self.log_colors.shape), # (batch_size,3,64,64,64)
            voxel_size=self._voxel_size
        )
        
        return self._renderer(cameras=cameras,volumes=volumes)[0]
    
# we now define a Huber loss function, which is robust to a small number of
# outliers from dragging the optimization away from the true optimal solution
# minimizing this loss function will move x closer to y
def huber(x,y,scaling=0.1):
    diff_sq = (x-y)**2
    loss = ((1+ diff_sq / (scaling ** 2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss

# put everything on the device
target_cameras = target_cameras.to(device)
target_images = target_images.to(device)
target_silhouettes = target_silhouettes.to(device)

# next we create an instance of our custom VolumeModel class
volume_size = 128
volume_model = VolumeModel(
    renderer,
    volume_size=[volume_size]*3,
    voxel_size=volume_extent_world/volume_size).to(device)

# optimizer
lr = 0.1
optimizer = torch.optim.Adam(volume_model.parameters(),lr=lr)
batch_size = 10
n_iter =  300

# main optimization loop. The densitites and colors of the volumes are rendered,
# and the resulting colors and silhouettes are compared with the observed multi-view
# images. The Huber loss between the rendered images and observed ground-truth images is
# minimized
for iteration in range(n_iter):
    if iteration == round(n_iter*0.75):
        print('Decreasing LR 10-fold ...')
        optimizer = torch.optim.Adam(volume_model.parameters(),lr=lr*0.1)
    optimizer.zero_grad()
    
    # only take 10 camera views at a time
    batch_idx = torch.randperm(len(target_cameras))[:batch_size]
    
    # sample the minibatch of cameras
    batch_cameras = FoVPerspectiveCameras(
        R=target_cameras.R[batch_idx],
        T=target_cameras.T[batch_idx],
        znear=target_cameras.znear[batch_idx],
        zfar=target_cameras.zfar[batch_idx],
        aspect_ratio=target_cameras.aspect_ratio[batch_idx],
        fov=target_cameras.fov[batch_idx],
        device=device
    )
    
    # render the scene from these camera views based on our 3D volume 
    # this call the forward pass, where volumetric rendering is peformed
    rendered_images, rendered_silhouettes = volume_model(batch_cameras).split([3,1],dim=-1)
    
    # compute silhouette error
    sil_error = huber(rendered_silhouettes[...,0], target_silhouettes[batch_idx]).abs().mean()
    
    # compute color error
    color_error = huber(rendered_images, target_images[batch_idx]).abs().mean()
    
    # compute loss
    loss = sil_error + color_error
    loss.backward()
    optimizer.step()
    
# for visualization after training
def generate_rotating_volume(volume_model, n_frames=50):
    logRs = torch.zeros(n_frames, 3, device=device)
    logRs[:, 1] = torch.linspace(0.0, 2.0 * 3.14, n_frames, device=device)
    Rs = so3_exp_map(logRs)
    Ts = torch.zeros(n_frames, 3, device=device)
    Ts[:, 2] = 2.7
    frames = []
    print('Generating rotating volume ...')
    for R, T in zip(Rs, Ts):
        camera = FoVPerspectiveCameras(
            R=R[None],
            T=T[None],
            znear=target_cameras.znear[0],
            zfar=target_cameras.zfar[0],
            aspect_ratio=target_cameras.aspect_ratio[0],
            fov=target_cameras.fov[0],
            device=device,
        )
        frames.append(volume_model(camera)[..., :3].clamp(0.0, 1.0))
    return torch.cat(frames)
    
# after optimization is finished, we can take the final resulting volumetric model and render
# from new angles
with torch.no_grad():
    rotating_volume_frames = generate_rotating_volume(volume_model, n_frames=7*4)
    
image_grid(rotating_volume_frames.clamp(0.,1.).cpu().numpy(),rows=4,cols=7,rgb=True,fill=True)
plt.savefig('rotating_volume.png')
plt.show()

