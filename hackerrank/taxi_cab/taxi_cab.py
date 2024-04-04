# Problem Statement -: A taxi can take multiple passengers to the railway station at the same time.
# On the way back to the starting point,the taxi driver may pick up additional passengers
# for his next trip to the airport.A map of passenger location has been created,represented as a square matrix.

# The Matrix is filled with cells,and each cell will have an initial value as follows:

# A value greater than or equal to zero represents a path.
# A value equal to 1 represents a passenger.
# A value equal to -1 represents an obstruction.
# The rules of motion of taxi are as follows:

# The Taxi driver starts at (0,0) and the railway station is at (n-1,n-1).
# Movement towards the railway station is right or down,through valid path cells.
# After reaching (n-1,n-1) the taxi driver travels back to (0,0) by travelling left
# or up through valid path cells. When passing through a path cell containing a
# passenger,the passenger is picked up.once the rider is picked up the cell
# becomes an empty path cell. If there is no valid path between (0,0) and (n-1,n-1),
# then no passenger can be picked. The goal is to collect as many passengers as
# possible so that the driver can maximize his earnings.

#  0 1
# -1 0
# ans = 1

# 0 0 0 1
# 1 0 0 0
# 0 0 0 0
# 0 0 0 0
# ans = 2

# 0 1 -1 
# 1 0 -1
# 1 1  1
# ans = 5

from typing import List

def num_passengers(grid: List[List[int]]) -> int:
    count = 0
    R=len(grid)
    C=R
    dp =[[0 for _ in range(C+1)] for _ in range(R+1)]
    dp[1][1] = 1
    
    dp2 =[[0 for _ in range(C+1)] for _ in range(R+1)]
    dp2[R-1][C-1] = 1
    
    # do it twice
    # 1st time, from start, go down and right
    # 2nd time , from end go up and left
    # If both DP matrices have a positive value
    # in the same position as a passenger, then
    # we can pick them up and we count that passenger
    # towards our total
    
    # so in first run
    for r in range(1,R+1):
        for c in range(1,C+1):
            if r == c and r == 1:
                continue
            # now update dp based on curr poss
            if grid[r-1][c-1] != -1:
                dp[r][c]=max(dp[r-1][c],dp[r][c-1])
            
    # 2nd run
    for r in range(R-1,-1,-1):
        for c in range(C-1,-1,-1):
            # print(f"r,c:{r},{c}")
            if r == c and r == R-1:
                continue
            # now update dp based on curr poss
            if grid[r][c] != -1:
                dp2[r][c]=max(dp2[r+1][c],dp2[r][c+1])

    print(f"dp1:{dp}")
    print(f"dp2:{dp2}")

    # check all places where dp is != 0 and grid has a vlaue of 1
    for r in range(R+1):
        for c in range(C+1):
            if dp[r][c]!=0 and dp2[r-1][c-1]!=0 and grid[r-1][c-1] == 1:
                count += 1

    return count

grid = [[0,1],
        [-1,0]]

print(f"count: {num_passengers(grid)}")