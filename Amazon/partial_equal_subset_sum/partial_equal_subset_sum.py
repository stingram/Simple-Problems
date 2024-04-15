# Given an array of positive numbers, determine if
# the array can be split such that the two partition
# sums are equal.


from typing import List

def partial_equal_subset_sum(nums: List[int]) -> bool:
    L = 0
    R = len(nums) - 1
    total = sum(nums)
    if total % 2 != 0:
        return False
    
    target = total // 2
    nums.sort()
    
    # compute combination of sums until we reach target
    sums = set([0])
    for num in nums:
        if num > target:
            break
        vals = set()
        for summ in sums:
            val = num + summ
            if val == target:
                return True
            elif val < target:
                vals.add(val)
        sums.update(vals)
            
    return False

nums = [1,5,11,5]
print(f"{partial_equal_subset_sum(nums)}")