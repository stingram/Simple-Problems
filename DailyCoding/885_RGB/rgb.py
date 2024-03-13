# Given an array of strictly the characters 'R', 'G', and 'B',
# segregate the values of the array so that all the Rs come first,
# the Gs come second, and the Bs come last. 
# You can only swap elements of the array.

# Do this in linear time and in-place.

# For example, given the array ['G', 'B', 'R', 'R', 'B', 'R', 'G'],
# it should become ['R', 'R', 'R', 'G', 'G', 'B', 'B']
from typing import List


def _sort_value(rgb_array,left,right,val):
    swap = left+1
    while left <= right and swap <=right:
        # if we can an 'R' we can advance pointers
        if rgb_array[left] == val:
            left += 1
            if left >= swap:
                swap = left + 1
            continue
        # we only got her because current r_pointer doesn't have an R
        if rgb_array[swap] == val:
            rgb_array[swap] = rgb_array[left]
            rgb_array[left] = val
            left += 1
        swap += 1
    return left

def sort_rgb(rgb_array: List[str]):
    # find i
    for i,val in enumerate(rgb_array):
        if val != 'R':
            break
    r_start = i

    for i in range(-1,-len(rgb_array)-1,-1):
        if rgb_array[i] != 'B':
            break
    b_start = i + len(rgb_array)

    print(f"B:{b_start}")
    
    left = _sort_value(rgb_array,r_start,b_start,'R')
    # now r_start is at first position where a 'G' can go
    # we just repeat process now
    _ = _sort_value(rgb_array,left,b_start,'G')

    return rgb_array


test = ['B','G','R']
print(f"{sort_rgb(test)}")


tests = [['R'],
         ['B','G','R'],
         ['G','G','G'],
         ['R','G','B','R','G','B']]

for test in tests:
    print(f"{sort_rgb(test)}")