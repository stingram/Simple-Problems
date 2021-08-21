import heapq
from typing import List
import math

class Solution:
    
    def _dist(self, point: List[int]):
        # don't need sqrt, because it won't change the ordering of the points
        return point[0]*point[0]+point[1]*point[1]
    
    def closest_points(self, k: int, points: List[List[int]]) -> List[List[int]]:
        '''
        O(k*lg(n)) - Time
        O(n) - Space
        '''
        # make heap array
        data = []
        
        for p in points:
            # compute distance and push with point to array
            data.append((self._dist(p),p))
        
        # heapify
        heapq.heapify(data)
        result = []
        
        # pop elements from heap
        for i in range(k):
            result.append(heapq.heappop(data)[1])
        
        return result
    
    def closest_points_slow(self, k: int, points: List[List[int]]) -> List[List[int]]:
        '''
        O(n*lg(n)) - Time
        O(n) - Space
        '''
        return sorted(points, lambda x: self._dist(x))[:k]
    
k = 3
points = [[-1,-1], [1,1], [2,2], [3,3],[4,4]]

print(Solution().closest_points(k, points))