from typing import List, Dict, Tuple


class Solution:
    
    
    def _valid_location(self, i,j):
        return i >= 0 and i < self.nrows and j >= 0 and j < self.ncols
    
    def _sink_island(self, graph, i, j):
        # check if this is valid location to sink
        if not self._valid_location(i,j):
            return
        
        # do bfs to find neighbors and sink them
        queue = [(i,j)]
        while queue:
            i,j = queue.pop(0)
            
            # sink this and then add neighbors to queue to sink
            graph[i][j] = -1
            if self._valid_location(i-1,j) and graph[i-1][j] == 1:
                queue.append((i-1,j))
            if self._valid_location(i+1,j) and graph[i+1][j] == 1:
                queue.append((i+1,j))
            if self._valid_location(i,j-1) and graph[i][j-1] == 1:
                queue.append((i,j-1))
            if self._valid_location(i,j+1) and graph[i][j+1] == 1:
                queue.append((i,j+1))    

    
    def _count_islands_util(self, graph: List[List[int]]):
        islands = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                # check if we are on an island
                # if on island, sink it
                if graph[i][j] == 1:
                    self._sink_island(graph,i,j)
                    # increment island count
                    islands+=1                    
        return islands
    
    
    def count_islands(self, graph: List[List[int]]) -> int:
        self.nrows = len(graph)
        self.ncols = len(graph[0])
        return self._count_islands_util(graph)
    
    
world = [[1,1,1,1,0],
         [1,1,0,1,0],
         [1,1,0,0,0],
         [0,0,0,0,0]]

print(f"Num islands: {Solution().count_islands(world)}")