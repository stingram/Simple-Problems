// # A unival tree (which stands for "universal value") is a tree where all nodes under it have the same value.
// # Given the root to a binary tree, count the number of unival subtrees.
// # For example, the following tree has 5 unival subtrees:

// #    0
// #   / \
// #  1   0
// #     / \
// #    1   0
// #   / \
// #  1   1


#include <iostream>
#include <tuple>

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

std::pair<bool,int> unival_helper(const Node* node)
{
    if(node == nullptr)
    {
        return {true, 0};
    }

    std::pair<bool,int> ret_left  = unival_helper(node->left);
    std::pair<bool,int> ret_right = unival_helper(node->right);

    if(node->left != nullptr and node->right != nullptr)
    {
        if(node->left->val == node->val && node->right->val == node->val && ret_left.first && ret_right.first)
        {
            return {true, 1 + ret_left.second + ret_right.second};
        }
    }
    else if(node->left == nullptr and node->right != nullptr)
    {
        if(node->right->val == node->val && ret_right.first)
        {
            return {true, 1+ ret_right.second};
        }
    }
    else if(node->left != nullptr and node->right == nullptr)
    {
        if(node->left->val == node->val && ret_left.first)
        {
            return {true, 1+ ret_left.second};
        }
    }
    else
    {
        return {true, 1};
    }

    return {false, ret_left.second + ret_right.second};

}


int unival(Node* root)
{
    return unival_helper(root).second;
}

int main()
{
    Node * root = new Node(0);

    root->left = new Node(1);
    root->right = new Node(0);

    root->right->left = new Node(1);
    root->right->right = new Node(0);

    root->right->left->left = new Node(1);
    root->right->left->right = new Node(1);

    std::cout << unival(root) << "\n";  // 5 

    return 0;
}