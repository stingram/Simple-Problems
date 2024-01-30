# Even though the loss is better compared to loss shown in the book, the final synthetic images
# of the cow don't look as good. For example, in the book the eyes are clearly rendered, while
# after running this script, they are not. 

# NOTE - for training to get going I changed initial learning rate from 1e-3 to 1e-4 

# we will train a nerf model without worrying about implementation details
import torch
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer
)

from utils.helper_functions import (
    generate_rotating_nerf,
    huber,
    sample_images_at_mc_locs,
    show_full_render
)

from nerf_model import NeuralRadianceField

if torch.cuda.is_available():
    device =torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
# these functions let us generate synthetic training data 
# and let us visualize images
from utils.plot_image_grid import image_grid
from utils.generate_cow_renders import generate_cow_renders

# with these utility functions we will generate camera angles, images
# and silhouettes of the synthetic cow from multiple different angles
target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=40,azimuth_range=180)
print(f"Generated {len(target_images)} images/silhouettes/cameras.")

# next we define a ray sampler of type MonteCarloRaysampler, which generates rays from a random subset of
# pixels from the image plane. We need a random sampler since we want to use mini-batch gradient descent algorithm
# to optimize the model. Importantly, the ray sampler samples points uniformly along the ray.
render_size = target_images[1]*2
volume_extent_world = 3.0
raysampler_mc = MonteCarloRaysampler(
    min_x = -1.0,
    max_x = 1.0,
    min_y = -1.0,
    max_y = 1.0,
    n_rays_per_image=750,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world
)

# next we define the ray marcher, uses volume densities and colors of points sampled along the ray to render the
# pixel value for that ray
raymarcher = EmissionAbsorptionRaymarcher()

# next we instantiate ImplicitRenderer which composes the ray sampler and ray marcher into a single data structure
renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher)

# while training, we'll want to visualize our progress, but to do that we'll need a systematic ray sampler
render_size = target_images.shape[1] * 2
volume_extent_world = 3.0
raysampler_grid = NDCMultinomialRaysampler(
    image_height=render_size,
    image_width=render_size,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world
)

# we'll now make an implicit renderer for our grid ray sampler
# note we can use the same ray marcher
renderer_grid = ImplicitRenderer(
    raysampler=raysampler_grid,raymarcher=raymarcher
)

# now we instantiate the nerf model
neural_radiance_field = NeuralRadianceField()

# move objects to the GPU
torch.manual_seed(1)

renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)

target_cameras = target_cameras.to(device)
target_images = target_images.to(device)
target_silhouettes = target_silhouettes.to(device)

neural_radiance_field = neural_radiance_field.to(device)

# hyperparameters
lr = 1e-4
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr)
batch_size = 6
n_iter = 3000

loss_history_color, loss_history_sil = [],[]

# training loop
for iteration in range(n_iter):
    if iteration == round(n_iter * 0.75):
        print('Decreasing LR 10-fold ...')
        optimizer = torch.optim.Adam(neural_radiance_field.parameters(),lr=lr*0.1)
    optimizer.zero_grad()
    batch_idx = torch.randperm(len(target_cameras))[:batch_size]
    
    batch_cameras = FoVPerspectiveCameras(
        R = target_cameras.R[batch_idx],
        T = target_cameras.T[batch_idx],
        znear = target_cameras.znear[batch_idx],
        zfar = target_cameras.zfar[batch_idx],
        aspect_ratio = target_cameras.aspect_ratio[batch_idx],
        fov = target_cameras.fov[batch_idx],
        device=device
    )
    
    # during each iteration, we obtained rendered pixels values and rendered silhouette values
    # from randomly sampled cameras using the NeRF model. These are predicted values from the
    # forward pass. We compare these to ground truth and compute loss as summation of huber loss
    # on the predicted colors and predicted silhouette.
    rendered_images_silhouettes, sampled_rays = renderer_mc(cameras=batch_cameras,
                                                            volumetric_function=neural_radiance_field)
    rendered_images, rendered_silhouettes = (rendered_images_silhouettes.split([3,1], dim=-1))
    
    # truth data
    silhouettes_at_rays = sample_images_at_mc_locs(target_silhouettes[batch_idx, ..., None], sampled_rays.xys)
    
    sil_err = huber(rendered_silhouettes,silhouettes_at_rays).abs().mean()
    
    # truth data
    colors_at_rays = sample_images_at_mc_locs(target_images[batch_idx],sampled_rays.xys)
    
    color_err =huber(rendered_images,colors_at_rays).abs().mean()
    
    # loss
    loss = color_err + sil_err
    loss_history_color.append(float(color_err))
    loss_history_sil.append(float(sil_err))
    
    loss.backward()
    optimizer.step()
    
    # we'll visualize the model performance after every 100 iterations
    # print(f"Iteration: {iteration}")
    if iteration % 100 == 0:
        show_idx = torch.randperm(len(target_cameras))[:1]
        fig = show_full_render(
            neural_radiance_field,
            FoVPerspectiveCameras(
                R=target_cameras.R[show_idx],
                T=target_cameras.T[show_idx],
                znear = target_cameras.znear[show_idx],
                zfar = target_cameras.zfar[show_idx],
                aspect_ratio=target_cameras.aspect_ratio[show_idx],
                fov=target_cameras.fov[show_idx],
                device=device
            ),
            target_images[show_idx][0],
            target_silhouettes[show_idx][0],
            renderer_grid,
            loss_history_color,
            loss_history_sil
        )
        fig.savefig(f"intermediate_{iteration}")

# after training, we take the final volumetric model and render images from new angles
with torch.no_grad():
    rotating_nerf_frames = generate_rotating_nerf(neural_radiance_field, target_cameras, renderer_grid, n_frames=3*5, device=device) 
image_grid(rotating_nerf_frames.clamp(0., 1.).cpu().numpy(), rows=3, cols=5, rgb=True, fill=True)
plt.show()
