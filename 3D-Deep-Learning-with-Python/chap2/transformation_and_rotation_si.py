import torch
from pytorch3d.transforms.so3 import so3_exp_map, so3_log_map, hat_inv, hat

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    print(f"WARNING: CPU only, this will be slow!")
    
log_rot = torch.zeros([4,3], device=device)

# We define a mini-batch of four rotations, each rotation is represented by
# one 3D vector. The direction of the vector represents the rotation axis and the 
# amplitude of the fector represents the angle of rotation
log_rot[0,0] = 0.001
log_rot[0,1] = 0.0001
log_rot[0,2] = 0.0002

log_rot[1,0] = 0.0001
log_rot[1,1] = 0.001
log_rot[1,2] = 0.0002

log_rot[2,0] = 0.0001
log_rot[2,1] = 0.0002
log_rot[2,2] = 0.001

log_rot[3,0] = 0.001
log_rot[3,1] = 0.002
log_rot[3,2] = 0.003

# we can use the hat operator to convert log_rot into 3x3 skew-symmetric matrices
log_rot_hat = hat(log_rot)
print(f'log_rot_hat shape: {log_rot_hat.shape}')
print(f'log_rot_hat: {log_rot_hat}')

# backward conversion from the skew-symmetric matrix form the 3D vector form is done
# with hat_inv operator
log_rot_copy = hat_inv(log_rot_hat)
print(f'log_rot_copy shape: {log_rot_copy.shape}')
print(f'log_rot_copy: {log_rot_copy}')

# from the gradient matrix, we can compute the rotation matrix by using
# so3_exp_map function
rotation_matrices = so3_exp_map(log_rot)
print(f"Rotation Matrices: {rotation_matrices}")

# To map rotation matrices back to the gradient matrix you can
# use so3_log_map
log_rot_again = so3_log_map(rotation_matrices)
print(f"log_rot_again:\n{log_rot_again}")