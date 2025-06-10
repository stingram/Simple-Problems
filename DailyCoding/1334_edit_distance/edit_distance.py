# The edit distance between two strings refers to the minimum
# number of character insertions, deletions, and substitutions
# required to change one string to the other. For example,
# the edit distance between “kitten” and “sitting” is three:
# substitute the “k” for “s”, substitute the “e” for “i”, and
# append a “g”.

# Given two strings, compute the edit distance between them.


def edit_distance(s1: str, s2: str) -> int:
    M = len(s1)
    N = len(s2)
    
    # "extra" row and col can thought of as empty character
    dp = [[0 for _ in range(N+1)] for _ in range(M+1)]
    
    # set 1st row
    for i in range(N+1):
        dp[0][i] = i
        
    # set 1st col
    for i in range(M+1):
        dp[i][0] = i
        
    # solve
    for i in range(1,M+1):
        for j in range(1,N+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    
    return dp[M][N]

# Time: O(N*M)
# Space: O(N*M)

s1 = "kitten"
s2 = "sitting"

print(f'Edit distance is : {edit_distance(s1, s2)}')
