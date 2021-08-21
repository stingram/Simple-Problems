from typing import List

class Solution(object):
    def getSingle(self, nums: List[int]) -> int:
        num_dict = {}
        for num in nums:
            num_dict[num] = num_dict.get(num,0) + 1
        for k, v in num_dict.items():
            if v == 1:
                return k        
        return 0
    
    def getSingleXOR(self, nums: List[int]) -> int:
        unique = 0
        for num in nums:
            unique ^= num
        return unique
    
print(Solution().getSingleXOR([1,2,3,3,2]))
        