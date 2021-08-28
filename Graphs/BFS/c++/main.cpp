#include <unordered_map>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_set>
#include <deque>


class Graph
{
    private:
        int V;
        std::unordered_map<int,std::vector<int>> g = {};


        void bfs_helper(std::vector<bool>& visited, const int s)
        {
            std::deque<int> my_queue;
            my_queue.push_back(s);
            visited[s] = true;
            while(my_queue.size() > 0)
            {
                int v = *(my_queue.begin());
                my_queue.erase(my_queue.begin());
                std::cout << "Node " << v << " processed.\n";
                for(int n: g[v])
                {
                    if(visited[n] == false)
                    {
                        my_queue.push_back(n);
                        visited[n] = true;
                    }
                }
            }
        }

    public:
    Graph()
    {
        this->V = 0;
    }
    void add_edge(const int a, const int b)
    {
        if(g.find(a) != g.end())
            add_node(a);
        if(g.find(b) != g.end())
            add_node(b);

        g[a].push_back(b);
    }
    void add_node(const int n)
    {
        if(g.find(n) != g.end())
        {
            return;
        }
        else
        {
            g[n] = {};
            V++;
        }
    }
    void bfs()
    {
        std::vector<bool> visited = std::vector<bool>(V, false);
        for(std::pair<int,std::vector<int>> element: g)
        {
            if(visited[element.first] == false)
            {
                bfs_helper(visited, element.first);
            }
        }
    }


    friend std::ostream& operator<< (std::ostream &out, Graph* const& graph)
    {
        for(std::pair<int,std::vector<int>> element: graph->g)
        {
            out << "Node: " << element.first << ", children: ";
            for(int num: element.second)
            {
                out << num << ", ";
            }
            out << "\n";
        }
        return out;
    }
};




int main()
{
    Graph* g = new Graph();
    g->add_node(0);
    g->add_node(1);
    g->add_node(2);

    g->add_edge(0,1);
    g->add_edge(0,2);

    std::cout << g << "\n";

    int start = 0;
    g->bfs();

    return 0;
}