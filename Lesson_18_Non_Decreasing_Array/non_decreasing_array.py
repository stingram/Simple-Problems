from typing import List

class Solution(object):
    def checkPossibility(self, nums: List[int]) -> bool:
        
        idx = None
        
        # go through array
        for i in range(len(nums) -1):
            if nums[i] > nums[i+1]:
                # make sure there's only 1 dip
                if idx is not None:
                    return False
                else:
                    idx = i
                    
        # only one dip, check all cases

        # if at beginning
        if idx == 0:
            return True
        
        # if at end - 1 = len(nums) - 2
        if idx == len(nums) -2:
            return True
        
        # if two points past idx is >= idx
        if nums[idx] <= nums[idx+2]:
            return True
        
        # if point before idx is <= point after idx
        if nums[idx-1] <= nums[idx+1]:
            return False
        
        # Anything else, return False
        return False