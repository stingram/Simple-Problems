from typing import List

class Solution(object):
    def find_missing(self, a: List[int]) -> int:
        vals = {}
        
        for num in a:
            if num > 0:
                vals[num] = 1
    
        for i in range(1,len(a)):
            if i not in vals: # O(1) operation
                return i
        return None
    
    
a = [-1,-2,5,4,2,1]
print(Solution().find_missing(a))
        