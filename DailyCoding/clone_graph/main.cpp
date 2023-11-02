#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>


class Node
{
    public:
    Node(int _val, std::vector<Node*> _neighbors)
    {
        val = _val;
        neighbors = _neighbors;
    }

    int val;
    std::vector<Node*> neighbors;
};


void print_graph(const Node* node)
{
    std::unordered_set<const Node*> visited;
    std::vector<const Node*> queue;
    queue.push_back(node);
    while(queue.size() > 0)
    {
        const Node* curr = queue.front();
        queue.erase(queue.begin());
        std::cout << "Node: " << curr->val << ". Neighbors: [";

        visited.insert(curr);
        for(auto neighbor: curr->neighbors)
        {
            std::cout << neighbor->val << ",";
            if(visited.find(neighbor) == visited.end())
            {
                queue.push_back(neighbor);
            }
        }
        std::cout << "].\n";
    }
}

Node* _clone_graph(const Node* node, std::unordered_map<const Node*,Node*>& visited)
{
    if(node == nullptr)
        return nullptr;

    if(visited.find(node) != visited.end())
    {
        return visited[node];
    }
    // create new node 
    Node* new_node = new Node(node->val, {});

    // add current node to visited
    visited[node] = new_node; 

    // add neighbors
    for(const Node* neighbor: node->neighbors)
    {
        new_node->neighbors.push_back(_clone_graph(neighbor,visited)); 
    }
    
    return visited[node];
}

Node* clone_graph(const Node* node)
{
    std::unordered_map<const Node*,Node*> visited = {};
    return _clone_graph(node, visited);
}



int main()
{
    Node* orig = new Node(1, {});
    Node* node2 = new Node(2, {});
    Node* node3 = new Node(3, {});
    orig->neighbors.push_back(node2);
    node2->neighbors.push_back(node3);
    node3->neighbors.push_back(orig);

    std::cout << "Orig Node: " << orig->val << ".\n";

    Node* new_node = clone_graph(orig);
    std::cout << "Original Graph:\n";
    print_graph(orig);
    std::cout << "Cloned Graph:\n";
    print_graph(new_node);


    return 0;
}