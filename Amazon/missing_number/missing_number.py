# you are given an array containing n distinct numbers,
# find the missing number

from typing import List
import numpy as np
def find_missing_number(arr: List[int]) -> int:
    n = len(arr)
    expected_sum = int(n*(n+1)/2)
    actual_sum = sum(arr)
    return expected_sum-actual_sum


#  1 to 20
x = np.arange(1,21,1,dtype=int)

# set 15 value (14-index) to 0
x[14] = 0
print(f"x:{x}")
# should print 15
print(f"Missing Number:{find_missing_number(x.tolist())}")
