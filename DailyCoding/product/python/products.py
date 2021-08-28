# Given an array of integers, return a new array such that each element at index i 
# of the new array is the product of all the numbers in the original array except the one at i.

# For example, if our input was [1, 2, 3, 4, 5], the expected output would be 
# [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would be [2, 3, 6].


# [1,2,3,4,5]
# [120,60,20,5,1] -> a
# [1,1,2,6,24] -> b
# prod[i] = a[i]*b[i]

from typing import List
class Solution(object):
    def products(self, nums: List[int]) -> List[int]:
        
        right = [1]*len(nums)
        left = [1]*len(nums)
        
        # build right -> O(n) time and space
        for n,i in zip(nums[::-1],range(len(nums)-2,-1,-1)):
            right[i] = right[i+1]*nums[i+1]

        # build left -> O(n) time and space
        for i,n in enumerate(nums):
            print(f"n: {n}")
            if i == 0:
                left[i] = 1
            else:
                left[i] = left[i-1]*nums[i-1]
        
        print(f"RIGHT: {right}.")
        print(f"LEFT: {left}.")
        
        # return result
        res = [1]*len(nums)
        for i in range(len(nums)):
            res[i] = right[i]*left[i]
        return res
            
            
nums = [1, 2, 3, 4, 5]
print(f"RES: {Solution().products(nums)}.")