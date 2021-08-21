#include <vector>
#include <unordered_map>
#include <functional>
#include <string>
#include <iostream>

class Node
{
    public:
    std::string val;
    Node* left;
    Node* right;
    Node(std::string val, Node* left=nullptr, Node* right=nullptr)
    {
        this->val=val;
        this->left=left;
        this->right=right;
    }
};




float arith_btree(Node*& node)
{
    if(node==nullptr)
    {
        return 0.0;
    }
    std::unordered_map<std::string, std::function<float(const float& a, const float& b)>> operations = 
    {
        {"+", [](const float& a, const float& b){return a+b;}},
        {"-", [](const float& a, const float& b){return a-b;}},
        {"*", [](const float& a, const float& b){return a*b;}},
        {"/", [](const float& a, const float& b){return a/b;}},
    };

    if(operations.find(node->val) != operations.end())
    {
        return operations[node->val](arith_btree(node->left), arith_btree(node->right));
    }
    else
    {
        return std::stof(node->val);
    }

    return 0.0;
}



int main()
{
    Node* node = new Node("*");
    node->left = new Node("+");
    node->right = new Node("+");
    node->left->left = new Node("3");
    node->left->right = new Node("2");
    node->right->left = new Node("4");
    node->right->right = new Node("5");

    std::cout << arith_btree(node) << "\n";
    
    
    return 0;
}