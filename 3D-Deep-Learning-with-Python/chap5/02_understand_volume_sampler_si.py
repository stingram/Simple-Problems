# Once we have ray bundle, we can use to do volume sampling. It is highly likely
# that the points defined our ray bundle don't fall exactly on a point in our 
# volume. We need to use an interpolation scheme to interpolate the densities and
# colors at points of ray from the densities and colors at volume.

import torch
from pytorch3d.structures import Volumes
from pytorch3d.renderer.implicit.renderer import VolumeSampler

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
# load our saved ray_bundle
checkpoint = torch.load('ray_sampling.pt')
ray_bundle = checkpoint.get('ray_bundle')

# now we can define a volume that has information about density and colors
# this requires making a densities tensor and colors tensor
# each volume will be a grid of 64x64x50
# we also will crate a batch of 10 volumes
image_size = 64
batch_size = 10
densities = torch.zeros([batch_size, 1, image_size, image_size, 50]).to(device)
colors = torch.zeros([batch_size, 3, image_size, image_size, 50]).to(device)
voxel_size = 0.1

volumes = Volumes(
    densities=densities,
    features=colors,
    voxel_size=voxel_size
)

# Now we need to define a VolumeSampler object based on our Volumes above. Here, we use
# bilinear interpolation for the volume sampling. The densities and colors of points on 
# the rays can then be easily obtained by passing ray_bundle to volume_sampler
volume_sampler = VolumeSampler(volumes=volumes, sample_mode="bilinear")
rays_densities, rays_features = volume_sampler(ray_bundle)

# Note we have one ray for each pixel and number of points on each ray is 50.
# Density is represented by 1 number and each colors needs 3 numbers for RGB
print(f"ray_densities shape = {rays_densities.shape}") # (10,64,64,50,1)
print(f"ray_features shape = {rays_features.shape}") # (10,64,64,50,3)

# finally we have these densities and colors to be used later
torch.save({
    'rays_densities': rays_densities,
    'rays_features': rays_features
}, 'volume_sampling.pt')


# debug
# print(f"{rays_densities[5,32,32,25,0]}")