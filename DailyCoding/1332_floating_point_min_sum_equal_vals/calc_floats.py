import numpy as np

# given an array of floating point values
# create list of integers call Y that
# The rounded sums of both arrays should be equal round(sum(x)) = sum(y)
# the sum of the absolute pairwise differences is minimized

# [1.3, 2.3, 4.4] -> [1, 2, 5]

# brute-force, try every combination
# is there any way past calculations can be meaningfully saved

from typing import List 
from math import floor, ceil
import numpy as np

def _helper(x,curr_y,best_y,curr_sum,min_diff, curr_diff, curr_ind, N):
    if curr_ind == N:
        return
    # process ind, we can either round up or round down

    # round up, continue
    new_y = ceil(x[curr_ind])
    curr_diff += abs(new_y - x[curr_ind])

    # if we're done we need to check if we meet constraints and
    # will need to save our list
    if curr_ind == N - 1:
        pass

    # round down, continue
    new_y = floor(x[curr_ind])
    curr_diff += abs(new_y - x[curr_ind])


def gen_y(x: List[float]):

    N = len(x)
    y = [floor(val) for val in x]

    # make pairs
    pairs = [(i,x_num-y_num) for i,(x_num,y_num) in enumerate(zip(x,y))]

    # sort by fractional size, keep track of inds
    pairs.sort(key= lambda p: -p[1])

    # get difference (should be an int), call it K
    # we want rounded sum of both arrays to be the same
    K = round(sum(x))-sum(y)
    print(f'K:{K}')

    # Note that if we had different constraints we would change the
    # calculation part slightly
    # | Constraint            | Formula for K                       | Sum after rounding |
    # | --------------------- | ----------------------------------- | ------------------ |
    # | **Match rounded sum** | `K = round(sum(x)) - sum(floor(x))` | Closest integer    |
    # | **Match floor**       | `K = floor(sum(x)) - sum(floor(x))` | ≤ original sum     |
    # | **Match ceil**        | `K = ceil(sum(x)) - sum(floor(x))`  | ≥ original sum     |


    # that number of ints, K, should be ceil, all others are floor
    # We already made items floor so we can just add 1 to items
    for i in range(K):
        y[pairs[i][0]] += 1

    # return y
    return y

T = 5
N = 5
for _ in range(T):
    x = np.random.uniform(1,10,(1,N))
    print(f'x:{x}\ny:{gen_y(list(x.squeeze()))}\n')