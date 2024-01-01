from typing import List
import sys

def max_score(target: List[int], dp: List[List[int]], l: int, r: int):
    # print(f"l,r: {l},{r}")
    if l > r:
        return 0
    
    if dp[l][r] != 0:
        return dp[l][r]

    for i in range(l,r+1): # need to figure out how to limit range here
        
        # compute point for hitting left interval
        left_points = max_score(target, dp, l, i-1)
        
        # compute point for hitting right interval
        right_points = max_score(target,dp,i+1,r)
        
        # left multiplier
        left_multiplier = 1 if l == 0 else target[l-1]
        
        # right multiplier
        right_multiplier = 1 if r == len(target) - 1 else target[r+1]
        
        val = left_multiplier * target[i] * right_multiplier
        
        dp[l][r] = max(dp[l][r], left_points + val + right_points)
        
    return dp[l][r]
        
def festival_game(target: List[int]) -> int: 
    # set up DP
    n = len(target)
    dp = [[0 for _ in range(n)] for _ in range(n)]
       
    return max_score(target,dp,0,n-1)

# if __name__ == '__main__':
#     target = [int(x) for x in input().split()]
#     res = festival_game(target)
#     print(res)


if __name__ == '__main__':
    # Check if there are enough command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py num1 num2 num3 ...")
        sys.exit(1)

    # Extract command-line arguments starting from the second position
    target = [int(x) for x in sys.argv[1:]]
    
    # Call your function with the provided list of integers
    res = festival_game(target)
    print(res)