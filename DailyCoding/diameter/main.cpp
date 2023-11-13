// Description: Given a binary tree, you need to compute the length of the diameter of the tree.
// The diameter of a binary tree is the length of the longest path between any two nodes in a tree.
// This path may or may not pass through the root.

//      1
//     / \
//    2   3
//   / \
//  4   5
// Ans: 4

#include <iostream>
#include <vector>

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

std::pair<int,int> diameterOfBinaryTreeHelper(TreeNode* node, int longest)
{
    // base case
    if(node == nullptr)
    {
        return {0,longest};
    }
    // base case, leaf node
    if(node->left == nullptr && node->right == nullptr)
    {
        if(1>longest)
        {
            longest = 1;
        }
        return {1,longest};
    }
    // get lengths of left and right sides
    std::pair<int,int> left = diameterOfBinaryTreeHelper(node->left,longest);
    std::pair<int,int> right = diameterOfBinaryTreeHelper(node->right,left.second);

    // update longest
    longest = std::max(left.first+right.first+1,longest);

    // return only longes side + 1
    return {std::max(left.first, right.first)+1,longest};
}


int diameterOfBinaryTree(TreeNode* root)
{
    return diameterOfBinaryTreeHelper(root,0).second;
}

int main()
{
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);
    // root->left->right->right = new TreeNode(6);
    std::cout << "Diameter of tree is: " << diameterOfBinaryTree(root) << ".\n";
    return 0;
}