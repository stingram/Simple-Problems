from typing import List

class Solution(object):
    def prod_wo_self(self, nums: List[int]):
        res = [1]*len(nums)
        
        # Get product of everything on the left of index
        for i in range(1, len(nums)):
            res[i] = res[i-1] * nums[i-1]
        
        # Begin accumulating products on the right of index
        R = 1
        for i in range(len(nums) -2, -1, -1):
            # Get updated running product
            R = R * nums[i+1]
            # Update result by multipling current value at index
            # by this new right product value
            res[i] = res[i] * R
            
        # return
        return res
    
input = [1, 2, 3, 4]    
print(Solution().prod_wo_self(input))
