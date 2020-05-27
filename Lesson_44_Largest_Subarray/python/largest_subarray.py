from typing import List

class Solution(object):
    def max_subarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        i=0
        j=len(nums)-1
        summation = sum(nums)
        max_summation = summation
        while(i<=j):
            # Get summation without i
            sum_right = summation-nums[i]
            
            # Get summation without j
            sum_left = summation-nums[j]
            
            # See which is smaller
            if sum_left < sum_right:
                # take right
                summation = sum_right
                i += 1
            else:
                summation = sum_left
                j -= 1
                
            # check if we are bigger than max_summation
            if summation > max_summation:
                max_summation = summation 
                                
        return max_summation
    
    def max_subarray_v2(self, nums):
        if len(nums) == 0:
            return 0
        res = nums[0]
        currMax = 0
        for n in nums:
            if currMax + n < 0:
                currMax = 0
                res = max(n, res)
            else:
                currMax += n
                res = max(currMax, res)
        return res
    
    
arr = [-2,1,-3,4,-1,2,1,-5,4]
print(Solution().max_subarray(arr))

arr = [4,-1,-4,5]
print(Solution().max_subarray(arr))