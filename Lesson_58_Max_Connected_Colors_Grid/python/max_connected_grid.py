class Grid:
    def __init__(self, grid):
        self.grid = grid
        self.y = len(grid) - 1
        self.x = len(grid[0]) - 1
        
    def get_size(self,y,x,r):
        
        r += 1
        self.grid[y][x] = 0
        
        if self.grid[y][max(0,x-1)] == 1:
            r = self.get_size(y,max(0,x-1),r)
        if self.grid[y][min(x+1,self.x)] == 1:
            r = self.get_size(y,min(x+1,self.x),r)
        if self.grid[max(0,y-1)][x] == 1:
            r = self.get_size(max(0,y-1),x,r)
        if self.grid[min(y+1,self.y)][x] == 1:
            r = self.get_size(min(y+1,self.y),x,r)
        return r
    
    def max_connected_colors(self) -> int:
        max_size = 0
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == 1:
                    s = self.get_size(i,j,0)
                    if s > max_size:
                        max_size = s
        return max_size
    
    
grid = [[1, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 1, 0, 0]]

print(Grid(grid).max_connected_colors())