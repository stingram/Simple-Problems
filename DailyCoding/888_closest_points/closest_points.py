from typing import List, Tuple

from heapq import heappop, heappush, heappushpop

def calc_dist(p1,p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def closest_points(points: List[Tuple[int,int]], cp: Tuple[int,int], k: int):
    res = [(0,0)]*k
    if k <= 0 or len(points)<1:
        return res
    heap = []
    for point in points:
        dist = calc_dist(point,cp)
        x = point[0]
        y = point[1]
        if len(heap) < k:
            heappush(heap,(-dist,x,y))
        else:
            heappushpop(heap,(-dist,x,y))
    for i in range(k):
        _, x, y = heappop(heap)
        res[-1-i]=(x,y)
        
    return res

points = [(0, 0), (5, 4), (3, 1)]
k = 2
cp = [1,2]
# [[0,0],[3,1]]
print(f"{closest_points(points,cp,k)}")