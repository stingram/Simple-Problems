# now that we have color and density values for all points sampled with the ray sampled,
# we now need to use to render the pixel value on the projected image

# The RGB value of each pixel is a weighted sum of the colors on the points of the corresponding ray
# The incident light intensity at each point of the ray is a product of (1-p_i), where p_i are the densities
# Given the probability that this point is occupied by a certain object is p_i, the expected light intensity
# reflected from this point is w_i = a p_i
# we just used w_i as the weights for the weighted sum of colors. Usually, we normalize the weights by applying
# softmax operation, such that the weights all sum to 1

import torch
from pytorch3d.renderer.implicit.raymarching import EmissionAbsorptionRaymarcher

# load densities and colors on rays from earlier
checkpoint = torch.load('volume_sampling.pt')
rays_densities = checkpoint.get('rays_densities')
rays_features = checkpoint.get('rays_features')

# we define ray_marcher and pass the densities and colors to ray_marcher. This gives us
# image_features, which are exactly rendered RGB values
ray_marcher = EmissionAbsorptionRaymarcher()
image_features = ray_marcher(rays_densities=rays_densities, rays_features=rays_features)

# we can now print image feature shape, which as we expect is [10,64,64,4]
# the outputs has 4 channels, the first three are RGB, the last channel is the alpha channel,
# which represents whether the pixel is in the foreground or background
print(f"image_features shape = {image_features.shape}")

