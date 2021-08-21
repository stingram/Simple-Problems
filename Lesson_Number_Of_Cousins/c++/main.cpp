#include <vector>
#include <iostream>
#include <string>
#include <utility>

class Node
{
    public:
        int val;
        Node* left;
        Node* right;

        Node(int val, Node* left=nullptr, Node* right=nullptr)
        {
            this->val=val;
            this->left=left;
            this->right=right;
        }
};



std::vector<int> find_cousins(Node*& node, int height, Node*& parent)
{
    if(height == 0){
        return {node->val};
    }
    if(node==parent || node==nullptr)
        return std::vector<int>();

    // return concatentation of left and right
    std::vector<int> l = find_cousins(node->left,height-1, parent);
    std::vector<int> r = find_cousins(node->right,height-1, parent);
    l.insert(l.end(), r.begin(), r.end());
    return l;
}



std::pair<int,Node*> find_node(Node*& node, Node*& target, Node** parent, int height=0)
{
    if(node==nullptr)
        return std::pair<int,Node*>(-1,nullptr);
    if(node==target)
        return std::pair<int,Node*>(height,*parent);
    std::pair<int,Node*> l = find_node(node->left, target, &node, height+1);
    if(std::get<0>(l) != -1)
        return l;
    else 
        return find_node(node->right, target, &node, height+1);
}

std::vector<int> list_cousins(Node*& root, Node*& target)
{
    std::vector<int> res = {};
    // get height and parent of target node
    std::pair<int,Node*> height_parent = find_node(root, target, nullptr);
    std::cout << "Height: " << std::get<0>(height_parent) << ", Parent val: " << std::get<1>(height_parent)->val << ".\n";

    // use height and parent to find cousins
    res = find_cousins(root, std::get<0>(height_parent),std::get<1>(height_parent));
    return res;
}

int main()
{
    Node* root = new Node(1);
    root->left = new Node(2);
    root->left->left = new Node(4);
    root->left->right = new Node(6);
    root->right = new Node(3);
    root->right->right = new Node(5);

    std::vector<int> cousins = list_cousins(root, root->right->right);
    for(int c: cousins)
    {
        std::cout << c << ", ";
    }
    std::cout << "\n";
    // [4, 6]

    return 0;
}

    
// #     1
// #    / \
// #   2   3
// #  / \    \
// # 4   6    5
