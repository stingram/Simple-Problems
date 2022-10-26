# This problem was asked by Airbnb.
# Given a list of integers, write a function
# that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.

# For example, [2, 4, 6, 2, 5] should return 13,
# since we pick 2, 6, and 5. [5, 1, 1, 5] should return 10, since we pick 5 and 5.

# Follow-up: Can you do this in O(N) time and constant space?

# Need two values as we traverse array:
# excl
# incl

# during first iteration:
# incl = arr[0]
# excl = 0

# during all subsequent iterations:
# temp = incl
# incl = max(incl, excl + arr[i])
# excl = temp
from typing import List
class Solution(object):
    def max_sum(self, nums: List[int])-> int:
        if len(nums) < 1:
            return 0 
        incl = nums[0]
        excl = 0
        
        for i in range(1,len(nums)):
            temp = incl
            incl = max(incl,excl+nums[i])
            excl = temp
            
        return max(incl,excl)

nums = [2,4,6,2,5]
print(f"Max sum: {Solution().max_sum(nums)}.") # 13

nums = [5,1,1,5]
print(f"Max sum: {Solution().max_sum(nums)}.") # 10