from typing import List
from collections import deque


def _get_neighbors(grid,visited,m,n,k,i,j,r):
    delta_r = [0,-1,0,1]
    delta_c = [1,0,-1,0]
    res = []
    
    for d_r, d_c in zip(delta_r, delta_c):
        n_r = i + d_r
        n_c = j + d_c
        if n_r >= 0 and n_r < m and n_c >= 0 and n_c < n:
            # this is a valid on the grid, but we need to check 
            # that the spot isn't 1, if it is, we need to make sure
            # we remove the obstacle
            if grid[n_r][n_c] == 0:
                res.append((n_r,n_c,r))
            elif grid[n_r][n_c] == 1 and r >= 1:
                res.append((n_r,n_c,r-1))
    return res

def shortest_path_with_elimination(grid: List[List[int]], k: int) -> int:
    """
    Return the minimum number of steps to reach bottom-right corner
    if you can eliminate up to k obstacles.
    """
    # can do bfs with extra state
    visited = set()
    m = len(grid)
    n = len(grid[0])
    
    # always start at 0,0,k
    q = deque([(0,0,k)])
    visited.add((0,0,k))
    num_steps = 0
    while q:
        qlen = len(q)
        for _ in range(qlen):
            # pop current state
            (i,j,r) = q.popleft()
            
            # check if we're done
            if i == m - 1 and j == n - 1:
                return num_steps
        
            # search neighbors
            for n_r,n_c,n_k in _get_neighbors(grid,visited,m,n,k,i,j,r):
                s = (n_r,n_c,n_k)
                if s not in visited:
                    print(f'Added state: {s}')
                    visited.add(s)
                    q.append(s)
                            
        num_steps += 1
    print(f'\n\n')
    return -1

import random

def run_tests():
    cases = [
        (
            [
                [0,0,0],
                [1,1,0],
                [0,0,0],
                [0,1,1],
                [0,0,0]
            ],
            1,
            6
        ),
        (
            [
                [0,1,1],
                [1,1,1],
                [1,0,0]
            ],
            1,
            -1
        ),
        (
            [[0]],
            0,
            0
        ),
        (
            [[0,1,0,0]],
            1,
            3
        ),
    ]

    for grid, k, want in cases:
        got = shortest_path_with_elimination(grid, k)
        assert got == want, f"Failed: {grid}, k={k} → {got}, want {want}"

    print("✅ All tests passed!")

if __name__ == "__main__":
    run_tests()