#include <vector>
#include <iostream>

// Description: Given a binary tree, flatten it to a linked list in-place.
// The linked list should be in the same order as a pre-order traversal of the binary tree.

//      1
//     / \
//    2   5
//   / \   \
//  3   4   6

// 1
//  \
//   2
//    \
//     3
//      \
//       4
//        \
//         5
//          \
//           6

class TreeNode
{
    public:
    TreeNode(int _val, TreeNode* _left = nullptr, TreeNode* _right = nullptr)
    {
        val = _val;
        left = _left;
        right = _right;
    }
    int val;
    TreeNode* left;
    TreeNode* right;
};

std::pair<TreeNode*,TreeNode*> flattenHelper(TreeNode* node)
{
    // base case - node is none
    if(node == nullptr)
        return {node,node};
    
    // base case - node has no children
    if(node->left == nullptr && node->right == nullptr)
        return {node,node};

    // link left
    std::pair<TreeNode*,TreeNode*> left = flattenHelper(node->left);

    // link right
    std::pair<TreeNode*,TreeNode*> right = flattenHelper(node->right);

    // rearrange
    TreeNode* temp = node->right;

    // if there are nodes on the left, link them first
    if(left.first != nullptr)
    {
        node->right = left.first;
        left.second->right = right.first;
    }

    // if there are no nodes on the left, just link right side
    else
    {
        node->right = right.first;
    }

    // return head and node
    return {node,right.second};
}


void flatten(TreeNode* node)
{
    flattenHelper(node);
    return;
}
    
int main()
{
TreeNode* root = new TreeNode(1);
root->left = new TreeNode(2);
root->left->left = new TreeNode(3);
root->left->right = new TreeNode(4);
root->right = new TreeNode(5);
root->right->right = new TreeNode(6);

flatten(root);

TreeNode* node = root;
while(node != nullptr)
{
    std::cout << node->val << "\n";
    node = node->right;
}

    return 0;
}