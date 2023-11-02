// world = [[1,1,1,1,0],
//          [1,1,0,1,0],
//          [1,1,0,0,0],
//          [0,0,0,0,0]]

// print(f"Num islands: {Solution().count_islands(world)}")


#include <vector>
#include <iostream>


void sink_island(std::vector<std::vector<int>>& world, int i, int j)
{
    // bfs
    int rows = world.size();
    int cols = world[0].size();

    if(i>=0 && i <rows && j>=0 && j< cols && world[i][j] == 1)
    {
        // sink this spot   
        world[i][j] = -1;
        // sink neighbors
        sink_island(world,i+1,j);
        sink_island(world,i-1,j);
        sink_island(world,i,j-1);
        sink_island(world,i,j+1);
    }


}


int count_islands(std::vector<std::vector<int>>& world)
{
    int rows = world.size();
    int cols = world[0].size();

    int num_islands = 0;
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            if(world[i][j] == 1)
            {
                sink_island(world,i,j);
                num_islands+=1;
            }
        }
    }

    return num_islands;
}


int main()
{
    std::vector<std::vector<int>> world = {{1,1,1,1,0},
                                           {1,1,0,1,0},
                                           {1,1,0,0,0},
                                           {0,0,0,0,0}};

    std::cout <<  "Num islands: " << count_islands(world) << ".\n";
    return 0;
}