# given n x m matrix of zeros and ones
# starting from top-left corner, how many ways to each bottom right?

# you can only move right and down
# 0 represents an empty space and 1 represents a wall

# assume 0 in start and 0 in end

# [[0,0,1]
#  [0,0,1]
#  [1,0,0]]

import numpy as np
from typing import Dict, List, Tuple

def _get_options(M: int, N :int, curr_pos: Tuple[int,int], matrix: List[List[int]]):
    options = []
    if curr_pos[0]+1 < M and matrix[curr_pos[0]+1][curr_pos[1]] != 1:
        options.append((curr_pos[0] + 1, curr_pos[1]))
    if curr_pos[1]+1 < N and matrix[curr_pos[0]][curr_pos[1]+1] != 1:
        options.append((curr_pos[0], curr_pos[1] + 1))
    return options

def _num_ways_helper(M,N,count,paths,curr_path,curr_pos, matrix):
    # base case
    if curr_pos == (M-1,N-1):
        count[0]+=1
        return
    # explore 
    for option in _get_options(M, N, curr_pos, matrix):
        _num_ways_helper(M,N,count,paths,curr_path,option, matrix)

def num_ways(matrix: List[List[int]]):
    M = len(matrix)
    N = len(matrix[0])
    count = [0]
    paths = []
    curr_path = [(0,0)]
    curr_pos = (0,0)
    _num_ways_helper(M,N,count,paths,curr_path,curr_pos, matrix)

    return count[0]


def _helper(matrix, M, N, curr, memo):
    if curr[0] == M - 1 and curr[1] == N - 1:
        memo[curr] = 1
        return

    if curr not in memo:
        memo[curr] = 0
        if curr[0] + 1 < M and matrix[curr[0] + 1][curr[1]] != 1:
            _helper(matrix, M, N, (curr[0]+1, curr[1]), memo)    
            memo[curr] += memo[(curr[0]+1,curr[1])]
        if curr[1] + 1 < N and matrix[curr[0]][curr[1] + 1] != 1:
            _helper(matrix, M, N, (curr[0], curr[1] + 1), memo)    
            memo[curr] += memo[(curr[0],curr[1] + 1)]

def num_ways_v2(matrix: List[List[int]]):
    M = len(matrix)
    N = len(matrix[0])
    memo: Dict[Tuple[int,int],int] = {}
    curr=(0,0)
    _helper(matrix,M, N, curr,memo)
    return memo[(0,0)]

# tests
SIZE = 4
for _ in range(10):
    M = np.random.randint(2,SIZE)
    x = np.random.random_integers(0,1,(M,M))
    x[0][0] = 0
    x[M-1][M-1] = 0
    print(f'There are {num_ways(x)} num ways and {num_ways_v2(x)} v2 to get from start to end for\n{x}.')


# Time is O(M x N) -> don't recompute anything
# Space is O(M X N) -> memo