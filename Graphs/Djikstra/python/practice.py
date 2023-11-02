from typing import List, Tuple
from collections import defaultdict
import heapq

class Solution:
    def __init__(self, times: List[Tuple[int,int,int]], n: int, k: int):
        return
    
    
# simple algorithm
def dijkstra(graph:List[List[int]], s: int) -> int:
    visited = set()
    min_heap = [(0,s)]
    prev = {}
    d = 0
    while min_heap:
        (d1, curr) = heapq.heappop(min_heap)
        if curr in visited:
            continue
        visited.add(curr)
        d += d1
        for n in graph[curr]:
            if n not in visited:
               heapq.heappush(min_heap,(d+graph[curr][n],n)) 
    return 0
    
def dijkstra_gen(graph: List[List[int]], s):
    visited = set()
    dists = [(0,s)]
    
    curr_d = 0
    while len(visited) < len(graph):
        (d, curr) = heapq.heappop((dists))
        if curr in visited:
            continue
        visited.add(curr)
        curr_d += d
        for n in graph[curr]:
            if n not in visited:
                heapq.heappush(dists,(curr_d+graph[curr][n],n))