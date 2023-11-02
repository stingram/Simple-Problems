# Description: You are climbing a staircase. It takes n steps to reach the top.
# Each time you can either climb 1 or 2 steps. 
# In how many distinct ways can you climb to the top?

from typing import List, Tuple, Dict

def num_ways(n: int) -> int:
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    
    counts = [0]*(n+1)
       
    # 1 steps
    counts[1] = 1
    
    # 2 steps
    counts[2] = 2
    
    for i in range(3,n+1):
            counts[i] = counts[i-1] + counts[i-2]
    
    return counts[-1]


n = 5
print(f"Number of way to climb {n} stairs is: {num_ways(n)}")