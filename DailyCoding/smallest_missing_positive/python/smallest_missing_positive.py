# Given an array of integers, find the first missing positive integer in linear time and constant space. 
# In other words, find the lowest positive integer that does not exist in the array. 
# The array can contain duplicates and negative numbers as well.

# For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0] should give 3.

# You can modify the input array in-place.
# O(n) time, O(1) space


from typing import List

class Solution(object):
    def find_smallest(self, nums: List[int]) -> int:
        # convert any negative numbers to 0
        for i in range(len(nums)):
            if nums[i] < 0:
                nums[i] = 0
                
        # mark values as negative if they are "in" the array
        for i in range(len(nums)):
            val = abs(nums[i])
            ind = val - 1
            if ind >= 0 and ind < len(nums):
                # now to that spot and read value
                num = nums[ind]
                # if we find a zero, we need to mark it negative in a way that doesn't
                # change the solution set. 
                if num == 0:
                    nums[ind] = -1*(len(nums)+1)
                # Make sure the number is negative
                else:
                    nums[ind] = -1*abs(nums[ind])
            else:
                pass
            
        # now loop through solution set until we get a non-negatice number
        for i in range(1,len(nums)+1):
            if nums[i-1] >= 0:
                return i
        return len(nums)+1
    
print(Solution().find_smallest([0,1,2]))            # 3
print(Solution().find_smallest([3, 4, -1, 1]))      # 2