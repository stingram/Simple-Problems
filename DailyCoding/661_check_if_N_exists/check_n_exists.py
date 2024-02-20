# This problem was asked by Netflix.

# Given a sorted list of integers of length N, determine
# if an element x is in the list without performing
# any multiplication, division, or bit-shift operations.

# Do this in O(log N) time.

# want half of a value
# we could use bit mask
# v
# [1,2,3,4,5], length = 5
# can we use sum of elements?
# we know the length of the list
# we know the sum of elements of the list
# if length is 5, mid is 2.
# start at 0 and end, then exponentially grow left and right
# once number has changes from being smaller/bigger than target,
# then search between this number and previous and repeat
# is that lg(n)?


def calc_mid(L,R):
    while R-L>1:
        L+=1
        R-=1
    return L

def is_element_in_list(nums, target):
    N = len(nums)
    L = 0
    R = N - 1
    while L<=R:
        mid = calc_mid(L,R)
        if nums[mid] == target:
            return True
        if nums[mid]>target:
            R=mid-1
        if nums[mid]<target:
            L=mid+1
    return False


nums=[1,2,3,4,5,6]
targets=[0,1,2,3,4,5,6]
# for target in targets:
#     print(f"Found: {is_element_in_list(nums,target)}")


# powers of 2: [1,2,4,8]
# start at 0
    

def find_idx_sorted(arr, x):
    powers_of_two = [1]
    while powers_of_two[-1] < len(arr):
        powers_of_two.append(powers_of_two[-1] + powers_of_two[-1])

    idx = 0
    # going from largest to smallest
    for pot in reversed(powers_of_two):
        # if index + power of 2 is in bounds and our target is bigger than
        # or equal to number at current index + power of 2, then increment
        # current index by power of 2
        if idx + pot < len(arr) and x >= arr[idx + pot]:
           idx += pot
    
    return idx

def contains_sorted(arr, x):
    return arr[find_idx_sorted(arr, x)] == x

nums=[1,2,3,4,5,6]
targets=[0,1,2,3,4,5,6]
for target in targets:
    print(f"Found: {is_element_in_list(nums,target)}")


# first we check
