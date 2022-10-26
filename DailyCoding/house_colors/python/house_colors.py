# A builder is looking to build a row of N houses that can be of K different colors. 
# He has a goal of minimizing cost while ensuring that no two neighboring houses are of the same color.

# Given an N by K matrix where the nth row and kth column represents the cost to build 
# the nth house with kth color, return the minimum cost which achieves this goal.
from typing import List


# Space is O(k), Time is O(N*k*k)

class Solution(object):
    
    def _min_prev_row(self, prev_row: List[int], curr_col: int) -> int:
        # since we are getting min cost from previous, we need to make sure we don't include cost
        # of the column for house we are currently building, since adjacent houses can't be the
        # same color
        del prev_row[curr_col]        
        return min(prev_row)
    
    def find_min_cost(self, ind_cost_matrix: List[List[int]]) -> int:
        # compute k
        k = len(ind_cost_matrix[0])
        
        # create two rows:  total_prev_costs, total_curr_costs
        total_prev_costs = ind_cost_matrix[0]
        total_curr_costs = [0]*k
        
        # start building total costs
        for i in range(1,len(ind_cost_matrix)):
            # compute current costs using previous
            for j in range(k):
                # each entry in the row, individual cost of current entry + minimum cost from previous total cost
                # using list() function to do deep copy so we can modify prev_costs without worry  
                total_curr_costs[j] = ind_cost_matrix[i][j] + self._min_prev_row(list(total_prev_costs),j)
                
            
            # DEBUG
            print(f"total_curr_costs: {total_curr_costs}")           
            
            # we have finished updating total curr_costs so now we overwrite total prev with total curr
            # we use list() because we need to overwrite contents, not just reference the total_curr_costs
            # with another variable 
            total_prev_costs = list(total_curr_costs)
            
        # return result
        return min(total_curr_costs)

# assume cost matrix is N x k, N is number of houses, k is number of colors
ind_cost_matrix = [[2,5,3],[1,6,2],[2,7,1],[4,3,3]]
print(f"{Solution().find_min_cost(ind_cost_matrix)}") # Should print 8


# Test reverse should be same result (different intermediate results, though)
print(f"{ind_cost_matrix[::-1]}")
print(f"{Solution().find_min_cost(ind_cost_matrix[::-1])}") # Should print 8