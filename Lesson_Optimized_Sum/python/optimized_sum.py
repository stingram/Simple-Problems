from typing import List
class ListFastSum(object):
    def __init__(self, nums: List):
        self.nums = nums
        self.acc = self._preprocess()
        
    def _preprocess(self):
        
        tsum = 0
        acc = [0]
        for i in range(len(self.nums)):
            tsum += self.nums[i]
            acc.append(tsum)
        return acc
    
    def opt_sum(self, ind1, ind2):
        return self.acc[ind2] - self.acc[ind1]
    
    
    
print(ListFastSum([1, 2, 3, 4, 5, 6, 7]).opt_sum(2, 5))
# 12