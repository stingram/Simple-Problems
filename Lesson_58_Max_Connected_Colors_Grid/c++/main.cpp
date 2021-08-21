#include <vector>
#include <iostream>
#include <cstdlib>
#include <numeric>
#include <algorithm>

class Grid
{
    public:
    std::vector<std::vector<int>> grid;
    int x;
    int y;
    Grid(std::vector<std::vector<int>>& in_grid)
    {
        this->grid.reserve(in_grid.size());
        for(int i=0;i<in_grid.size();i++)
        {
            this->grid.push_back(in_grid[i]);
        }
        this->y = this->grid.size() - 1;
        this->x = this->grid[0].size() - 1;
    }

    int get_size(int y, int x, int r)
    {
        r += 1;
        this->grid[y][x] = 0;
        if(this->grid[y][std::max(0,x-1)] == 1)
             r = get_size(y,std::max(0,x-1),r);
        if(this->grid[y][std::min(x+1,this->x)] == 1)
             r = get_size(y,std::min(x+1,this->x),r);
        if(this->grid[std::max(0,y-1)][x] == 1)
             r = get_size(std::max(0,y-1),x,r);
        if(this->grid[std::min(y+1,this->y)][x] == 1)
             r = get_size(std::min(y+1,this->y),x,r);
        return r;
    }

    int max_connected_colors()
    {
        int max_size = 0;
        for(int i=0;i<=this->y;i++)
        {
            for(int j=0;j<=this->x;j++)
            {
                if(this->grid[i][j] == 1)
                {
                    int s = get_size(i,j,0);
                    if(s > max_size)
                    {
                        max_size = s;
                    }
                }
            }
        }
        return max_size;
    }

};

int main()
{
    std::vector<std::vector<int>> grid = {{1,0,0,1},
                                          {1,1,1,1},
                                          {0,1,0,0}};

    std::cout << Grid(grid).max_connected_colors() << "\n";
}

// class Grid:
//     def __init__(self, grid):
//         self.grid = grid
//         self.y = len(grid) - 1
//         self.x = len(grid[0]) - 1
        
//     def get_size(self,y,x,r):
        
//         r += 1
//         self.grid[y][x] = 0
        
//         if self.grid[y][max(0,x-1)] == 1:
//             r = self.get_size(y,max(0,x-1),r)
//         if self.grid[y][min(x+1,self.x)] == 1:
//             r = self.get_size(y,min(x+1,self.x),r)
//         if self.grid[max(0,y-1)][x] == 1:
//             r = self.get_size(max(0,y-1),x,r)
//         if self.grid[min(y+1,self.y)][x] == 1:
//             r = self.get_size(min(y+1,self.y),x,r)
//         return r
    
//     def max_connected_colors(self) -> int:
//         max_size = 0
//         for i in range(len(self.grid)):
//             for j in range(len(self.grid[0])):
//                 if self.grid[i][j] == 1:
//                     s = self.get_size(i,j,0)
//                     if s > max_size:
//                         max_size = s
//         return max_size
    
    
// grid = [[1, 0, 0, 1],
//         [1, 1, 1, 1],
//         [0, 1, 0, 0]]

// print(Grid(grid).max_connected_colors())