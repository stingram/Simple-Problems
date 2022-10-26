# O(N) in space and time for setup
# O(1) in space and time for sum calculation

from typing import List
class Solution:
    def __init__(self, in_list:List[int]):
        self.sums = [0] # added for convenience
        for i, val in enumerate(in_list):
            self.sums.append(self.sums[i]+val)
        print(f"Sums right: {self.sums}.")
     
       
    def sum(self, i: int, j: int):
        if i >= j:
            return None
        # do other checks
        
        return self.sums[j] - self.sums[i]
    
# L = [1, 2, 3, 4, 5]
# sum(1, 3)
 
print(f"Sum: {Solution([1, 2, 3, 4, 5]).sum(1,3)}") # 5
print(f"Sum: {Solution([1, 2, 3, 4, 5]).sum(1,4)}") # 9  