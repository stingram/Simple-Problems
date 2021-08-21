#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <iostream>





int maze_paths(std::vector<std::vector<int>>& maze)
{
    int r = maze.size();
    int c = maze[0].size(); 

    std::vector<std::vector<int>> paths(r, std::vector<int>(c,0));


    paths[0][0] = 1;
    int left = 0;
    int top = 0;

    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            // top corner
            if(i==0 && j ==0)
            {
                continue;
            }
            // check if value at position is 1
            if(maze[i][j] == 1)
            {
                continue;
            }
            // top row only add from left
            if(i==0)
            {
                paths[i][j] = paths[i][j-1];
            }
            // left column only add from top
            else if(j==0)
            {
                paths[i][j] = paths[i-1][j];
            }
            else{
                left = paths[i][j-1];
                top = paths[i-1][j];
                paths[i][j] = left+top;
            }
        }
    }

    return paths[r-1][c-1];
}






int main()
{
    std::vector<std::vector<int>> maze = {{0,1,0},
                                          {0,0,1},
                                          {0,0,0}};

    std::cout << maze_paths(maze) << "\n";
    return 0;
}