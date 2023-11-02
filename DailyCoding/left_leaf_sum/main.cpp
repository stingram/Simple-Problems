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


int sumOfLeftLeavesHelper(TreeNode* node, int sum, bool could_be_left_leaf)
{
    // base case
    if(node == nullptr)
        return sum;
    
    // if the node has no children
    if(node->left == nullptr && node->right == nullptr)
    {
        if(could_be_left_leaf)
        {
            return sum+node->val;
        }
        else
        {
            return sum;
        }
    }

    // if it does have children, update sum from children
    sum = sumOfLeftLeavesHelper(node->left,sum,true);
    sum = sumOfLeftLeavesHelper(node->right,sum,false);

    // return sum
    return sum;

}


int sumOfLeftLeaves(TreeNode* node)
{
    return sumOfLeftLeavesHelper(node,0,false);
}

//       3
//      / \
//     9  20
//        / \
//       15  7
//  Ans: 9 + 15 = 24

int main()
{
    TreeNode* root = new TreeNode(3);
    root->left = new TreeNode(9);
    // root->left->left = new TreeNode(-16);
    root->right = new TreeNode(20);
    root->right->left = new TreeNode(15);
    root->right->right = new TreeNode(7);

    std::cout << "Sum of left leaves is: " << sumOfLeftLeaves(root) << ".\n";

    return 0;
}