#include <vector>
#include <cstdlib>
#include <string>
#include <iterator>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <sstream>


class Node
{
    public:
    int value;
    Node* left;
    Node* right;

    Node(int val)
    {
        value = val;
        left = nullptr;
        right = nullptr;
    }
};

class Solution
{
    public:
    std::string serialize(const Node* node)
    {
        if(node==nullptr)
        {
            return "#";
        }
        return std::to_string(node->value) + " " + serialize(node->left) + " " + serialize(node->right); 
    }

    Node* helper(std::stringstream& values)
    {
        std::string val;
        values >> val;
        
        if(val == "#")
        {
            return nullptr;
        }
        Node* node = new Node(std::stoi(val));
        node->left = helper(values);
        node->right = helper(values);
        return node;
    }

    Node* deserialize(std::string serial)
    {

        std::stringstream values(serial);
        return helper(values);
    }
};

int main()
{
    Node* root = new Node(1);
    root->left = new Node(3);
    root->right = new Node(4);
    root->left->left = new Node(2);
    root->left->right = new Node(5);
    root->right->right = new Node(7);

    std::string out =  Solution().serialize(root);

    std::cout << out << "\n";

    Node* res_node = Solution().deserialize(out);
    std::cout << std::to_string(res_node->right->right->value) << "\n";

    return 0;
}