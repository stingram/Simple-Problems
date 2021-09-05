#include <sstream>
#include <iostream>
#include <string>
#include <vector>


class Node
{
    public:
    int val;
    Node* left;
    Node* right;
    Node(int val, Node* left = nullptr, Node* right = nullptr)
    {
        this->val = val;
        this->left = left;
        this->right = right;
    }
};

class Solution
{
    public:
    std::string serialize(const Node* node)
    {
        if(node == nullptr)
            return "#";
        return std::to_string(node->val) + " " + serialize(node->left) + " " + serialize(node->right);
    }
    Node* helper(std::stringstream& values)
    {
        std::string val;
        values >> val;
        if(val == "#")
            return nullptr;
        Node* node = new Node(std::stoi(val));
        node->left = helper(values);
        node->right = helper(values);
        return node;
    }

    Node* deserialize(const std::string val_string)
    {
        std::stringstream values(val_string);
        return helper(values);
    }
};


int main()
{
    Node* root = new Node(1);
    root->left= new Node(2);
    root->right = new Node(3);
    root->left->left = new Node(4);
    root->left->right = new Node(5);
    root->right->left = new Node(6);
    root->right->right = new Node(7);

    std::string serial_res = Solution().serialize(root);
    std::cout << serial_res << "\n";

    root = Solution().deserialize(serial_res);

    serial_res = Solution().serialize(root);
    std::cout << serial_res << "\n";

    return 0;
}