# Given an array Z of 0s and 1s, divide the array into 3 non-empty parts,
# such that all of these parts represent the same binary value.

# If it is possible, return any [i, j], such that

# Z[0], Z[1], ..., Z[i] is the first part;
# Z[i+1], Z[i+2], ..., Z[j-1] is the second part, and
# Z[j], Z[j+1], ..., Z[Z.length - 1] is the third part.
# All three parts have equal binary value.
# If it's not possible return [-1, -1]

from typing import List


def find_needed_index(in_array, start_index, required_ones, required_zeros):
    # consume 1s first
    num_ones_seen = 0
    for i,val in enumerate(in_array[start_index:]):
        if val == 1:
            num_ones_seen += 1
        if num_ones_seen >= required_ones:
            break
        
    # we need to now "consume" zeros
    num_zeros_seen = 0
    for j in range(i+start_index,len(in_array[start_index:])):
        if in_array[j] == 0:
            num_zeros_seen += 1            
        if num_zeros_seen >= required_zeros:
            break
    
    # j is our I index
    return j


def compute_value(in_array,start_index,end_index):
    val = 0
    power = 0
    for ind in range(end_index,start_index,-1): 
        if in_array[ind] == 1:
            val += 2**power
        power+=1
    return val

def binary_string_split(in_array: List[int]):
    # first count number of 1s
    ones = 0
    for val in in_array:
        if val == 1:
            ones += 1
    
    if ones == 0:
        return[0,0]
    
    # check number of 1s is divisible by 3
    if ones % 3 != 0:
        return [-1,-1]
    # get required ones
    required_ones = ones // 3
    
    # now we also need required zeros from the right
    required_zeros = 0
    for i in range(-1,-len(in_array)-1,-1):
        if in_array[i] == 1:
            break
        else:
            required_zeros += 1
            
    # find i by advancing pointer past required_ones and required zeros
    i = find_needed_index(in_array, 0, required_ones, required_zeros)
    
    # find j, same as finding i, but just starting at later index
    j = find_needed_index(in_array, i+1, required_ones, required_zeros)+1
        
    # compute value of range 0,i,i+1 to j-1,j to end
    left = compute_value(in_array,0-1,i)
    mid = compute_value(in_array,i,j-1)
    right = compute_value(in_array,j-1,len(in_array)-1)
    if left == mid and left == right:
        return [i,j]
    else:
        return [-1,-1]
    
# tests
in_array = [1,0,1,0,1] # [0,3]
print(binary_string_split(in_array))

in_array = [1,1,0,1,1,0,1,1,0] # [2,6]
print(binary_string_split(in_array))

in_array = [1,1,0,0,1,1,0,0,1,1] # [1,6]
print(binary_string_split(in_array))