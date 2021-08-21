class Solution(object):
    def _permute_helper(self, nums, start=0):
        if start == len(nums):
            return nums[:]
        
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            self._permute_helper(nums, start+1)
    
    
    def permute(self, nums):
        return self._permute_helper(nums)
    
print(Solution().permute([1,2,3]))