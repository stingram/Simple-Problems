# given a stream of integers, output median at every element
from typing import List
from heapq import heappop, heappush
def median_stream(nums: List[int]) -> List[int]:
    low_heap = []
    high_heap = []

    res = []

    for num in nums:
        heappush(low_heap,-num)
        heappush(high_heap, -heappop(low_heap))

        if high_heap>low_heap:
            heappush(low_heap,-heappop(high_heap))

        if len(low_heap) == len(high_heap):
            res.append((-low_heap[0]+high_heap[0])/2.0)
        else:
            res.append(-low_heap[0])
        
    return res