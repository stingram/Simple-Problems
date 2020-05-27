from collections import defaultdict
# heap in python is a min heap
import heapq
class Solution:
    def topKFrequent(self, nums, k):
        count = defaultdict(int)
        for n in nums:
            count[n] +=1
        heap = []
        
        for key, v in count.items():
            heapq.heappush(heap, (v,key))
            if len(heap) > k:
                heapq.heappop(heap)
        res = []
        while len(heap) > 0:
            res.append(heapq.heappop(heap)[1])
        return res
    
input = [1,1,1,2,2,3]
k = 1

print(Solution().topKFrequent(input, k))