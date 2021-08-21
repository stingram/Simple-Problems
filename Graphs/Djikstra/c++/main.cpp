#include <vector>
#include <iostream>
#include <unordered_map>
#include <limits>
#include <numeric>

class Graph
{
    public:
        std::vector<std::vector<int>> graph;
        int V;

        Graph(int v)
        {
            V = v;
        }

        int min_node(std::unordered_map<int,bool>& visited, std::vector<int>& dists)
        {
            int min_dist = std::numeric_limits<int>::max();
            int min_node = 0;
            for(int i=0;i<V;i++)
            {
                if(visited.find(i)==visited.end())
                {
                    if(dists[i]< min_dist)
                    {
                        min_dist = dists[i];
                        min_node = i;
                    }
                }
            }
            return min_node;
        }

        void print_info(std::unordered_map<int,bool>& visited, std::vector<int>& dists, std::unordered_map<int,int>& prev)
        {
            std::cout << "Vertex \tDistance from Source\n";
            for(int node=0; node<V; node++)
            {
                std::cout << node << ", \t " << dists[node] << "\n";
            }
                

            std::cout << "Vertex \tParent\n";
            for(int node=0; node<V; node++)
            {
                std::cout << node << "\t " << prev[node] << "\n"; 
            }
        }

        void dijkstra(int n)
        {
            // initialize data structures
            std::unordered_map<int,int> prev;
            std::unordered_map<int,bool> visited;
            std::vector<int> dists(V, std::numeric_limits<int>::max());

            dists[n] = 0;
            while(visited.size()<V)
            {
                int curr = min_node(visited, dists);
                for(int n=0;n<V;n++)
                {
                    // for all nodes not visited that are connected to curr
                    if(visited.find(n)==visited.end() && graph[curr][n] != 0)
                    {
                        //get distance
                        int dist_n = dists[curr]+graph[curr][n];

                        // if distnce is smaller than before, update dists
                        if(dist_n < dists[n])
                        {
                            dists[n] = dist_n;
                            prev[n] = curr;
                        }
                    }
                }
                // update visited
                visited[curr] = true;
            }

            print_info(visited, dists, prev);

            return;
        }
};




// Driver program
int main()
{
    Graph g = Graph(9);

    g.graph = {{0, 4, 0, 0, 0, 0, 0, 8, 0},
            {4, 0, 8, 0, 0, 0, 0, 11, 0},
            {0, 8, 0, 7, 0, 4, 0, 0, 2},
            {0, 0, 7, 0, 9, 14, 0, 0, 0},
            {0, 0, 0, 9, 0, 10, 0, 0, 0},
            {0, 0, 4, 14, 10, 0, 2, 0, 0},
            {0, 0, 0, 0, 0, 2, 0, 1, 6},
            {8, 11, 0, 0, 0, 0, 1, 0, 7},
            {0, 0, 2, 0, 0, 0, 6, 7, 0}
            };
    g.dijkstra(0);
    return 0;
}


