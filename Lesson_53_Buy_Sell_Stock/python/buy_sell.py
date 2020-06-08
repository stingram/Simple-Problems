from typing import List

class Solution:
    def buy_sell(self, prices: List[int]) -> int:
        if not prices:
            return 0
        
        res = 0
        max_val = 0
        
        # loop through prices backward
        for i in range(len(prices)-1, -1, -1):
            # Compute max
            max_val = max(max_val, prices[i])
            
            # subtract current val from this max
            profit = max_val - prices[i]
            
            # if this is greater than current return, update return
            res = max(res, profit)
            
        return res
    
    
prices = [7,1,5,3,6,4]
print(Solution().buy_sell(prices))

prices = [7,6,4,3,1]
print(Solution().buy_sell(prices))