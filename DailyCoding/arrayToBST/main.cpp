// Description: Given an integer array nums sorted in non-decreasing order,
// convert it to a height-balanced binary search tree (BST).

// A height-balanced BST is defined as a binary tree in which the 
// depth of the two subtrees of every node never differs by more than one.

#include <vector>
#include <iostream>

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


TreeNode* sortedArrayToBST(const std::vector<int>& nums)
{
    // base case
    if(nums.size() == 0)
        return nullptr;
    
    // get split position
    int node_pos = nums.size()/2;

    // create node
    TreeNode* node = new TreeNode(nums[node_pos]);

    // Create left side of tree
    const std::vector<int>& left_subset = {nums.begin(),nums.begin()+node_pos};
    node->left = sortedArrayToBST(left_subset);

    // Create right side of tree
    const std::vector<int>& right_subset = {nums.begin()+node_pos+1,nums.end()};
    node->right = sortedArrayToBST(right_subset);

    // return the node 
    return node;
}

int main()
{
    std::vector<int> arr = {1,2,3,4,5};
    TreeNode* root = sortedArrayToBST(arr);
    std::cout << "Root: " << root->val << ".\n";
    std::cout << "Root.left: " << root->left->val << ".\n";
    std::cout << "Root.right: " << root->right->val << ".\n";   
    std::cout << "Root.left.left: " << root->left->left->val << ".\n";
    std::cout << "Root.right.left: " << root->right->left->val << ".\n";

    return 0;
}