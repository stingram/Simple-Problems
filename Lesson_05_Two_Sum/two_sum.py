class Solution:
    def twoSum(self, nums, target):
        my_dict = {}
        for i, num in enumerate(nums):
            # check if the complement for this number exists in the dictionary
            if target - num in my_dict:
                # if it does, return the complement index and this number's index
                return [my_dict[target - num], i]
            # Add this number as a key and it's index to our dictionary
            my_dict[num] = i
        return "N"
    
    
# Need to return the indices in the nums list that add to target
# assume a unique solution
class Solution:
     def two_sum(self, nums: List, target: int):
         my_dict = {}
         for i, num in enumerate(nums):
             if target-num in my_dict:
                 return [my_dict[target-num], i]
             my_dict[num] = i
         return False


class Solution:
    def two_sum(self, nums: List, target: int):
        my_dict = {}
        for i, num in enumerate(nums):
            if target-num in my_dict:
                return [my_dict[target-num], i]
            my_dict[num] = i
        return False
