#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <queue>

class Node
{
    public:
    std::vector<std::string> children;
    Node(std::string parent, std::string child)
    {
        children.push_back(child);
    }
};

class Graph
{
    public:
    std::unordered_map<std::string,std::vector<std::string>> g;
    int V;
    Graph(int v)
    {
        g = {};
        V = v;
    }
    
    void add_edge(const std::string& parent, const std::string& child)
    {
        if(g.find(parent) != g.end())
        {
            g[parent].push_back(child);
        }
        else
        {
            g[parent] = {child};
        }
    }
    std::vector<std::string> top_sort(void)
    {
        // BUild in-degree 
        std::unordered_map<std::string,int> in_degree;
        std::queue<std::string> g_queue;
        std::vector<std::string> sorted;
        for(auto& v: g)
        {
            for(auto s: v.second)
            {
                if(in_degree.find(s)!= in_degree.end())
                {
                    in_degree[s]+=1;
                }
                else
                {
                    in_degree[s] = 1;
                }
            }
        }
        // add all nodes that aren't in in_degree to our queue
        for(auto& v: g)
        {
            if(in_degree.find(v.first) == in_degree.end())
            {
                g_queue.push(v.first);
            }
        }


        int cnt = 0;
        // add any node that have in_degree = 0 to queue
        while(g_queue.empty()==false)
        {
            std::string n = g_queue.front();
            g_queue.pop(); 
            sorted.push_back(n);

            for(auto s: g[n])
            {
                in_degree[s] -= 1;
                if(in_degree[s] == 0)
                {
                    g_queue.push(s);
                }
            }
            cnt++;
        }
        if(cnt!=V)
        {
            return {""};
        }
        else
        {
            return sorted;
        }

    }

};

int main()
{
    Graph g = Graph(3);
    g.add_edge("A", "B");
    g.add_edge("B", "C");
    g.add_edge("A", "C");
    std::vector<std::string> sorted = g.top_sort();
    for(auto s: sorted)
    {
        std::cout << s << ", ";
    }
    std::cout << "\n";

    g = Graph(5);
    g.add_edge("A", "C");
    g.add_edge("A", "B");
    g.add_edge("C", "B");
    g.add_edge("B", "D");
    g.add_edge("C", "D");
    g.add_edge("C", "E");

    sorted = g.top_sort();
    for(auto s: sorted)
    {
        std::cout << s << ", ";
    }
    std::cout << "\n";
    return 0;

}






// g = Graph(3)
// g.add_edge("A", "B")
// g.add_edge("B", "C")
// g.add_edge("A", "C")

// print(top_sort(g))


// g = Graph(5)
// g.add_edge("A", "C")
// g.add_edge("A", "B")
// g.add_edge("C", "B")
// g.add_edge("B", "D")
// g.add_edge("C", "D")
// g.add_edge("C", "E")

// print(top_sort(g))