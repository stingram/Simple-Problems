#include <vector>
#include <iostream>

class TreeNode
{
    public:
    TreeNode(int _val, TreeNode* _left = nullptr, TreeNode* _right = nullptr)
    {
        val  =_val;
        left = _left;
        right = _right;
    }
    int val;
    TreeNode* left;
    TreeNode* right;
};

std::pair<int,int> robHelper(TreeNode* node, int odd_sum, int even_sum, int level)
{
    if(node == nullptr)
        return {odd_sum, even_sum};
    
    if(level % 2 == 0)
    {
        even_sum += node->val;
    }
    else
    {
        odd_sum += node->val;
    }

    // for left
    std::pair<int,int> sums = robHelper(node->left, odd_sum, even_sum,level+1);

    // for right
    sums = robHelper(node->right,sums.first, sums.second, level+1);

    return sums;
}

int rob(TreeNode* node)
{
    std::pair<int,int> sums = robHelper(node,0,0,0);
    return std::max(sums.first, sums.second);
}

int main()
{
//      3
//     / \
//    2   3
//     \   \ 
//      3   1

// Ans: 7 

TreeNode* root = new TreeNode(3);
root->left = new TreeNode(2);
root->right = new TreeNode(3);
root->left->right = new TreeNode(3);
root->right->right = new TreeNode(1);

std::cout << "Max sum that can be robbed is: " << rob(root) << ".\n";
    return 0;
}