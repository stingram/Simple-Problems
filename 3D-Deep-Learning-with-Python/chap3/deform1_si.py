import os
import sys
import torch
from pytorch3d.io import load_ply, save_ply
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency
)

import numpy as np
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print(f"WARNING: CPU only, this will be slow!")
    
# read in vertexes and faces, since it's a point cloud,
# faces is an empty tensor
verts, faces = load_ply("pedestrian.ply")
verts = verts.to(device)
faces = faces.to(device)

# Perform some normalization and change the tensor shapes for later processing
center = verts.mean(0)
verts = verts - center

# scale is single scalar value, which is the largest vertex value
print(f"verts.abs().max(): {verts.abs().max()}.") 
print(f"verts.abs().max(0): {verts.abs().max(0)}.")
print(f"verts.abs().max(0)[0]: {verts.abs().max(0)[0]}.") # max vertex value for each column, ie, 3 values total, 1 for x, 1 for y, 1 for z
scale = max(verts.abs().max(0)[0]) # the largest of the 3 values are chosen
print(f"Scale: {scale}.\nScale Shape: {scale.shape}")

verts = verts / scale # so now max distance from 0 in any dimension is 1
print(f"Verts Shape: {verts.shape}")

verts = verts[None,:,:]
print(f"New Verts Shape: {verts.shape}")

# create a mesh representing a sphere with ico_sphere function
# this src_mesh will be our optimization variable. It will start as a sphere
# and be optimized to fit the point cloud
# In this case, there will be 2562 vertices
src_mesh = ico_sphere(4,device)

# next, we want to define deform_verts variable. This is a tensor of vertex displacements,
# where for each vertex in src_mesh, there is a displacement. We are going to optimize
# deform_verts for finding the optimal deformable mesh
src_vert = src_mesh.verts_list()
print(f"src_vert len: {len(src_vert)}.")
print(f"src_vert[0] shape: {src_vert[0].shape}.")

deform_verts = torch.full(src_vert[0].shape,0.0,device=device,requires_grad=True)
print(f"Deform Verts shape: {deform_verts.shape}.")


# define SGD optimizer
optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

# weights for loss components
w_chamfer = 1.0
w_edge = 1.0
w_normal = 0.01
w_laplacian = 0.1

for i in range(2000):
    print(f"Iteration number = {i}")
    optimizer.zero_grad()
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    sample_trg = verts
    sample_src = sample_points_from_meshes(new_src_mesh, verts.shape[1]) # sample only as many verts as there are in the point cloud we are trying to fit
    
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    loss_edge = mesh_edge_loss(new_src_mesh)
    loss_normal = mesh_normal_consistency(new_src_mesh)
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    
    loss = loss_chamfer*w_chamfer + loss_edge*w_edge + loss_normal*w_normal + loss_laplacian*w_laplacian
    loss.backward()
    optimizer.step()
    
# now we extract obtained vertices and faces
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

# undo normalization 
final_verts = final_verts*scale + center

# Save the result
final_obj = os.path.join("./", "deform1_si.ply")
save_ply(final_obj, final_verts, final_faces, ascii=True)