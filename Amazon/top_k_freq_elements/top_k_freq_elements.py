# Given an integer array nums and an integer k, 
# return the k most frequent elements. You may
# return the answer in any order.

from typing import List
from collections import defaultdict
from heapq import heappush, heappop, heappushpop

def top_k_freq_elements(nums: List[int], k: int) -> List[int]:
    res = [0]*k
    
    if len(nums) <= k:
        return nums
    
    # create counter
    freqs = defaultdict(int)
    for num in nums:
        freqs[num] += 1
    
    # push to heap
    heap = []
    for num,count in freqs.items():
        if len(heap)<k:
            heappush(heap,(count,num))
        else:
            heappushpop(heap, (count,num))

    # pop k items
    for i, (_,num) in enumerate(heap):
        res[i] = num
    
    return res

nums = [1,1,1,2,2,3,3,3,3,3,5,4]
k = 2
print(f"{top_k_freq_elements(nums,k)}")