You are given an array of n integers and a number k. Return indices of the two numbers such that they add up to exactly k. For example, given the array [2, 3, 15] and k = 5, the answer is [0,1]

# k could be any integer
# array values could be any integer and there can be duplicates

# any length array

# no ordering to the array

# if no sum, return an empty list


# Assume no duplicates
[2, 3, 2, 3, 15] and k = 5

# Solution 1
# scan through array once and build a dictionary with {value:ind}
# then I can loop over the dictionary checking for each value if the k - value is also in the dictionary
# I can remove both values from the dictionary when I do get a match





# alternative
# in place sort the array
# looping over the array
# starting at the beginning, select an element do a binary search for the k - element value
# stop when element is larger than k

# I check my output list for this index - back to O(n^2) time

# k = 7

# [2,3,4,5,6]



# Alternative number 3
# sorted array
# start with Left pointer = 0, Right pointer = end
# while Left < Right
#      check array[L] + array[R]
#      if sum < k:
       #    L++
       # if sum > k:
       #    R--
       # if sum == k:
       #    append to result list L,R
       #    L++
       #    R--
    
    
    
    
    
    
    
    
    
# Solution 1
# scan through array once and build a dictionary with {value:ind}
# then I can loop over the dictionary checking for each value if the k - value is also in the dictionary
# I can remove both values from the dictionary when I do get a match

from typing import List
class Solution(object):
    def find_indices_for_sum(self, nums: List[int], k:int) -> List[List[int]]:
        
        # Results
        res = []
        
        # Build dictionary of value, index pairs
        nums_dict = {}
        for i, num in enumerate(nums):
            if num not in nums_dict:
                nums_dict[num] = i
                
        # check if k - nums exists
        for i,num in enumerate(nums):
            if k - num in nums_dict:
                
                # add to result
                res.append([nums_dict[num],nums_dict[k-num]])
                
                # prevent duplicates
                nums_dict.pop(num)
                nums_dict.pop(k-num)
        
        return res
    
    
arr = [2,3,4,5,6]
k = 7
print(Solution().find_indices_for_sum(arr,k))
