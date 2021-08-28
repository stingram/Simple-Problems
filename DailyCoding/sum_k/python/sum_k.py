# Given a list of numbers and a number k, return whether any two numbers from the list add up to k.
# For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.
# Bonus: Can you do this in one pass?
from typing import List

class Solution(object):
    def __init__(self):
        pass
    
    def find_sum(self, arr: List[int], k: int) -> bool:
        comps = {}
        for i,num in enumerate(arr):
            if num in comps:
                return True
            if k - num not in comps:
                comps[k-num] = i
        return False
    
    

arr = [10,15,3,3,7]
k = 6

print(Solution().find_sum(arr,k))