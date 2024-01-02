import open3d
import os
import torch

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures.meshes import join_meshes_as_batch

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print(f"WARNING: CPU only, this will be slow!")
    
# load meshes
mesh_names = ['cube.obj', 'diamond.obj', 'dodecahedron.obj']
data_path = './data'

for mesh_name in mesh_names:
    mesh = open3d.io.read_triangle_mesh(os.path.join(data_path,mesh_name))
    # commented out so we don't have to close each mesh window
    # open3d.visualization.draw_geometries([mesh], 
    #                                      mesh_show_wireframe=True,
    #                                      mesh_show_back_face=True)
    
# build list of meshes with pytorch3d
mesh_list = list()
device = torch.device('cuda')
for mesh_name in mesh_names:
    mesh = load_objs_as_meshes([os.path.join(data_path,mesh_name)], device=device)
    mesh_list.append(mesh)
    
# create mini-batch
mesh_batch = join_meshes_as_batch(mesh_list,include_textures=False)

# show list format of the batch
vertex_list = mesh_batch.verts_list()
print(f"Vertex List: {vertex_list}")
face_list = mesh_batch.faces_list()
print(f"Face List: {face_list}")

# show padded format of the batch
vertex_padded = mesh_batch.verts_padded()
print(f"Vertex Padded: {vertex_padded}")
face_padded = mesh_batch.faces_padded()
print(f"Face Padded: {face_padded}")

# show packed format of the batch
vertex_packed = mesh_batch.verts_packed()
print(f"Vertex Packed: {vertex_packed}")
num_vertices = vertex_packed.shape[0]
face_packed = mesh_batch.faces_packed()
print(f"Face Packed: {face_packed}")

mesh_batch_noisy = mesh_batch.clone()

motion_gt = np.array([3,4,5])
motion_gt = torch.as_tensor(motion_gt)
print(f"Motion Ground Truth: {motion_gt}, shape: {motion_gt.shape}")

# adds dimension to front. # (3) -> (1,3)
motion_gt = motion_gt[None,:]
motion_gt = motion_gt.to(device)
print(f"Motion Ground Truth: {motion_gt}, shape: {motion_gt.shape}")

# simulates a noisy depth camera by generating gaussian noise
# with a mean equal to motion_gt. The noises are added using
# the offsets_verts function
noise = (0.1**0.5)*torch.randn(mesh_batch_noisy.verts_packed().shape).to(device)
motion_gt = np.array([3,4,5])
motion_gt = torch.as_tensor(motion_gt).to(device)
noise = noise + motion_gt
mesh_batch_noisy = mesh_batch_noisy.offset_verts(noise).detach()

# now we create tensor for motion estimate
motion_estimate = torch.zeros(motion_gt.shape, device=device, requires_grad=True)

# create optimizer
optimizer = torch.optim.SGD([motion_estimate], lr=0.1, momentum=0.9)

# do SGD
for i in range(200):
    optimizer.zero_grad()
    current_mesh_batch = mesh_batch.offset_verts(motion_estimate.repeat(num_vertices,1))
    
    sample_trg = sample_points_from_meshes(current_mesh_batch,5000)
    sample_src = sample_points_from_meshes(mesh_batch_noisy,5000)
    
    loss, _ = chamfer_distance(sample_trg,sample_src)
    loss.backward()
    optimizer.step()
    print(f"i:{i}, motion_estimate: {motion_estimate}")