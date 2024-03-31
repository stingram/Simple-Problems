# You have an N by N board. Write a function that, given N,
# returns the number of possible arrangements of the board where
# N queens can be placed on the board without threatening each
# other, i.e. no two queens share the same row, column, or diagonal.

from typing import List, Set

def place_queen(state,r,c,cols_used):
    # check if any queen is in diagonal with another with proposed queen
    for qr, qc in enumerate(state):
        if qr != r and qc != c:
            if qc != -1 and abs(qr-r) == abs(qc-c):
                return False
    # if we get here we can place the queen
    state[r]=c
    cols_used.add(c)
    return True

def remove_queen(state,r,cols_used):
    cols_used.remove(state[r])
    state[r] = -1
    return

def _helper(state: List[int], row: int, n: int, solutions_count: List[int],cols_used: Set[int]):
    if row == n:
        solutions_count[0] += 1
        return
    for i in range(n):
        if i not in cols_used: # makes sure we don't use a column with a queen already in it
            if place_queen(state,row,i,cols_used):
                _helper(state,row+1,n,solutions_count,cols_used)
                remove_queen(state,row,cols_used)

def num_queen_arrangements(n: int) -> int:
    row = 0
    state = [-1]*n # index is row, value represents column used
    solutions_count = [0]
    cols_used = set()
    _helper(state,row,n,solutions_count,cols_used)
    return solutions_count[0]

tests = [1,2,3,4,5,6,7,8]
for test in tests:
    print(f"Num ways to place {test} queens is: {num_queen_arrangements(test)}")