#!/bin/python3

import math
import os
import random
import re
import sys
import numpy as np

# Complete the 'conv2d_arbitrary_dims' function below.
#
# The function is expected to return a 3D_FLOAT_ARRAY.
# The function accepts following parameters:
#  1. 3D_FLOAT_ARRAY img
#  2. 4D_INTEGER_ARRAY offset_map
#  3. 4D_FLOAT_ARRAY weights
#

# time efficiency is O((img_height-kernel_height)*(img_width-kernel_width)*(out_channels)*(in_channels)*(kernel_height*kernel_width))


# Reuse from previous problem
def dot_product(img,offset_map,weights,i,j) -> float:
    # i,j is center of window
    # print(f"offset_map:{offset_map},{offset_map.shape}")
    # print(f"weights:{weights},{weights.shape}")
    
    # compute "regular" offsets
    
    reuglar_offsets = []
    for r in range(weights.shape[0]):
        for c in range(weights.shape[1]):
            reuglar_offsets.append((r+i, c+j))
    # this is wrong, in first test case the length is 9, but it should be 6 
    # print(f"reg_offses_len:{len(reuglar_offsets)}")
    
    # offset for image
    offsets=[]
    for r in range(offset_map.shape[0]):
        for c in range(offset_map.shape[1]):
            offsets.append((offset_map[r][c][0], offset_map[r][c][1]))

    # combine offsets
    for i, regular in enumerate(reuglar_offsets):
        offsets[i]=(offsets[i][0]+regular[0], offsets[i][1]+regular[1])
    
    # compute dot product
    res = 0
    weights = weights.flatten()
    for offset, weight in zip(offsets, weights):
        if 0<=offset[0]<img.shape[0] and 0<=offset[1]<img.shape[1]:
            res += img[offset[0]][offset[1]]*weight
    return res


def channel_dot_product(img, offset_map, weights, i, j, out_c, in_C):
    # loop over all input channels
    val = 0
    for in_c in range(in_C):
        val += dot_product(img[:, :, in_c], offset_map[:,:,in_c,:], weights[:, :, in_c, out_c], i, j)
    return val

def conv2d_arbitrary_dims(img, offset_map, weights):
    # print(f"OFF:{offset_map.shape}")
    
    # assume that output shape isn't changed just because
    # we have offsets, so if a sample is out of bounds, it doesn't
    # contribute to sum
    out_H = img.shape[0]-weights.shape[0]+1
    out_W = img.shape[1]-weights.shape[1]+1
    
    in_C = weights.shape[2]
    out_C = weights.shape[3]
    
    out = np.zeros((out_H, out_W, out_C))
    print(f"weight_shape:{weights.shape}")
    print(f"out_shape:{out.shape}")

    f_H = weights.shape[0]
    f_W = weights.shape[1]
    
    for i in range(out_H):
        for j in range(out_W):
            for c in range(out_C):
                out[i][j][c] = channel_dot_product(img, offset_map, weights, i, j, c, in_C)
                
    return out

def read_pair_of_ints(pair_as_text):
    values = pair_as_text.split(',')
    pair = [(int(values[0]), int(values[1]))]
    return pair

if __name__ == '__main__':
    # fptr = open(os.environ['OUTPUT_PATH'], 'w')
    fptr = open("out.txt", 'w')

    k_0 = int(input().rstrip())
    k_1 = int(input().rstrip())
    num_rows = int(input().rstrip())
    num_cols = int(input().rstrip())
    num_input_channels = int(input().rstrip())
    num_output_channels = int(input().rstrip())

    np.random.seed(42)
    img = np.random.rand(num_rows, num_cols, num_input_channels)
    offset_map = np.random.randint(-5, 6, (k_0, k_1, num_input_channels, 2))
    weights = (4.*np.random.rand(k_0, k_1, num_input_channels, num_output_channels))-2

    result = conv2d_arbitrary_dims(img, offset_map, weights)

    fptr.write('\n'.join([' '.join(map(str, np.around(x, 3))) for x in result]))
    fptr.write('\n')
    fptr.close()
