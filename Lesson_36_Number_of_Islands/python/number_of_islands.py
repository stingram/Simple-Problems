class Solution(object):
    
    
    
    def _get_neighbors(self, i, j):
        top = (max(i-1,0),j)
        bottom = (min(i+1,len(grid)),j)
        left = (i,max(j-1,0))
        right = (i, min(j+1,len(grid[0])))
        
        return [top, bottom, left, right]    
    
    
    def num_islands(self, grid):
    
        # set labeled
        labeled = {}
        
        # set num islands
        num_islands = 0
        
        # loop through grid
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                
                # if we a zero we ignore
                if grid[i][j] == 0:
                    pass
                
                # if we get a 1 we check neighbors and if it's neighbors
                # have a label
                else:
                    neighbors = self._get_neighbors(i,j)
                    for n in neighbors:
                        if n in labeled:
                            labeled[(i,j)] = labeled[n]
                    # Got here so it's a new island
                    if labeled.get((i,j)) is None:
                        num_islands += 1
                        labeled[(i,j)] = num_islands         
        # return
        return num_islands
    
    
    
    def num_islands_v2(self, grid):
        def sink_island(grid,r,c):
            if grid[r][c] == '1':
                grid[r][c] = '0'
            else:
                return
            if r + 1 < len(grid):
                sink_island(grid, r+1, c)
            if r - 1 >= 0:
                sink_island(grid, r-1, c)
            if c + 1 < len(grid[0]):
                sink_island(grid, r, c+1)
            if c - 1 >= 0:
                sink_island(grid, r, c-1)
        counter = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    counter += 1
                    sink_island(grid, i, j)
        return counter            
        
        





grid = [[1,1,1,1,0],
        [1,1,0,1,0],
        [1,1,0,0,0],
        [0,0,0,0,0]]

grid = [[1,1,0,0,0],
        [1,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,1]]

print(Solution().num_islands(grid))