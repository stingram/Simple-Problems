# Given an array of integers, return an array of triplets (in any order)
# such that i != j != k and nums[i] + nums[j] + nums[k] = 0. Note that
# the solution set must not include duplicate triplets (i.e., [1, 0, 0]
# and [0, 1, 0] are duplicative).

from typing import List

def three_sum(nums: List[int]) -> List[List[int]]:
    target = 0
    res = []
    
    # sort nums
    nums.sort()
    
    for i in range(len(nums)-2):  # iterate until third last element for L
        # skip duplicate elements
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i+1, len(nums)-1 # two pointers
        while left<right:
            curr_sum = nums[i]+nums[left]+nums[right]
            
            if curr_sum == target:
                res.append([nums[i],nums[left],nums[right]])
                # add one to left, subtract one from right
                # could just update one, but this is slightly
                # more efficient
                left += 1
                right -= 1
                
                # update to skip over duplicates since we found target
                while left<right and nums[left-1] == nums[left]:
                    left += 1
                while left<right and nums[right] == nums[right-1]:
                    right -= 1
            elif curr_sum > target:
                right -= 1 # dont' have to worry about duplicates since we didn't hit target
            else:
                left += 1   

    return res


nums = [-1,0,1,2,-1,-4]
print(f"{three_sum(nums)}")