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

R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
# print(f"Camera center: {camera_center}")

# now we can generate an image, image_ref from this camera position
silhouette = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)
image_ref = phong_renderer(meshes_world=teapot_mesh,R=R, T=T)

silhouette = silhouette.cpu().numpy()
image_ref = image_ref.cpu().numpy()

plt.figure(figsize=(10,10))
plt.imshow(silhouette.squeeze()[...,3]) # only plot the alpha channel of the RGBA image
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'target_silhouette_si.png'))
plt.close()

plt.figure(figsize=(10,10))
plt.imshow(image_ref.squeeze())
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'target_rgb_si.png'))
plt.close()


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([3.0, 6.9, 2.5],dtype=np.float32)).to(meshes.device))
        
    def forward(self):
        R = look_at_rotation(self.camera_position[None,:], device=self.device) # creates shape [1,3,3]
        T = -torch.bmm(R.transpose(1,2), self.camera_position[None, :, None])[:,:,0] # shape: [1,3]
        
        image = self.renderer(meshes_world=self.meshes.clone(),R=R,T=T)
        loss = torch.sum((image[...,3] - self.image_ref)**2)
        
        return loss, image
    

model = Model(meshes=teapot_mesh, renderer=silhouette_renderer, image_ref=image_ref).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

_, image_init = model()
plt.figure(figsize=(10,10))
plt.imshow(image_init.detach().squeeze().cpu().numpy()[...,3])
plt.grid(False)
plt.title('Starting Silhouette')
plt.savefig(os.path.join(output_dir, 'starting_silhouette_si.png'))
plt.close()

# During each optimization iteration, we will save the rendered image from the 
# camera position to output directory
for i in range(200):
    if i % 10 == 0:
        print(f"i: {i}")
    optimizer.zero_grad()
    loss, _ = model() # we can just call model here because we gave inputs at it's instantiation
    loss.backward()
    optimizer.step()
    
    if loss.item() < 500:
        break
    
    R = look_at_rotation(model.camera_position[None, :],device=model.device)
    T = -torch.bmm(R.transpose(1,2), model.camera_position[None, :, None])[:,:,0] # Size will be (1,3)
    
    
    image = phong_renderer(meshes_world=model.meshes.clone(),R=R,T=T)
    image = image[0,...,:3].detach().squeeze().cpu().numpy()
    image = img_as_ubyte(image)
    
    plt.figure()
    plt.imshow(image[..., :3])
    plt.title(f"iter: {i:d}, loss: {loss.data:0.2f}.")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "fitting_" + str(i) + "_si.png"))
    plt.close()
    