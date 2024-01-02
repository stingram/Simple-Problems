import open3d
import os
import sys
import torch

import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer
)

from pytorch3d.renderer.mesh.shader import HardPhongShader

sys.path.append(os.path.abspath(''))

# check cow mesh
#Load meshes and visualize it with Open3D
# mesh_file = "./data/cow_mesh/cow.obj"
# print('visualizing the mesh using open3D')
# mesh = open3d.io.read_triangle_mesh(mesh_file)
# open3d.visualization.draw_geometries([mesh],
#                                      mesh_show_wireframe = True,
#                                      mesh_show_back_face = True,
#                                      )

DATA_DIR = "./data"
obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")
device = torch.device('cuda')
mesh = load_objs_as_meshes([obj_filename], device=device)

# This changes where the object is relative to the camera
# So instead of 2.7 for the distance, a distance of 80 made the
# cow very tiny
R,T = look_at_view_transform(2.7,0,180)

# camera always at 0,0,0?
cameras = PerspectiveCameras(device=device,R=R,T=T)

lights = PointLights(device=device, location=[[0.0,0.0,-3.0]])

raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0,
    faces_per_pixel=1,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader = HardPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)
# here images is a list of images, where each element is [H,W,C]
images = renderer(mesh)
plt.figure(figsize=(10,10))
# Since we only want to display 1 image, we select the first element, and take all H,W elements and
# only the first three channels, since we want RGB, equivalent to images[0][:,:,:3]
# plt.imshow(images[0][:,:,:3].cpu().numpy())

plt.imshow(images[0,...,:3].cpu().numpy())
plt.axis("off")
plt.savefig('lights_at_front_si.png')
plt.show()

# puts the light behind the cow, so now we only see the effect of the ambient light
# None adds a dimension so it's shape is [1,3]
# print(f"Lights shape: {torch.tensor([0.0,0.0, 1.0],device=device)[None].shape}")
lights.location = torch.tensor([0.0,0.0, 1.0],device=device)[None]
images = renderer(mesh,lights=lights)

plt.figure(figsize=(10,10))
plt.imshow(images[0,...,:3].cpu().numpy())
plt.axis("off")
plt.savefig('lights_at_back_si.png')
plt.show()

# now we set the materials in the renderer to have almost no ambience
materials = Materials(
    device=device,
    specular_color=[[0.0,1.0,0.0]],
    shininess=10.0,
    ambient_color=((0.01,0.01,0.01),),
)
images = renderer(mesh, lights=lights, materials=materials)
plt.figure(figsize=(10,10))
plt.imshow(images[0,...,:3].cpu().numpy())
plt.axis("off")
plt.savefig('dark_si.png')
plt.show()

# now we rotate the camera. Not that shininess parameter is is the "p"
# parameter in the Phong lighting model. specular_color of [0,1,0] implies
# the surface is shiny in the green component

R,T = look_at_view_transform(dist=2.7,elev=10,azim=-150)
cameras = PerspectiveCameras(device=device,R=R,T=T)
lights.location = torch.tensor([[2.0,2.0,-2.0]],device=device)
materials = Materials(
    device=device,
    specular_color=[[0.0,1.0,0.0]],
    shininess=10.0
    )

images = renderer(mesh,lights=lights,materials=materials,cameras=cameras)
plt.figure(figsize=(10,10))
plt.imshow(images[0,...,:3].cpu().numpy())
plt.axis("off")
plt.savefig('green_si.png')
plt.show()

# now we change the specular color to red and increase shininess
materials = Materials(
    device=device,
    specular_color=[[1.0, 0.0, 0.0]],
    shininess=20.0
    )
images = renderer(mesh, lights=lights, cameras=cameras, materials=materials)
plt.figure(figsize=(10,10))
plt.imshow(images[0,...,:3].cpu().numpy())
plt.axis("off")
plt.savefig('red_si.png')
plt.show()

# now we turn off the shininess
materials = Materials(
    device=device,
    specular_color=[[0.0,0.0,0.0]],
    shininess=0.0
    )

images = renderer(mesh,lights=lights,cameras=cameras,materials=materials)
plt.figure(figsize=(10,10))
plt.imshow(images[0,...,:3].cpu().numpy())
plt.axis("off")
plt.savefig('blue_si.png')
plt.show()