#include <vector>
#include <limits>
#include <iostream>
#include <algorithm>

class TreeNode{
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

std::pair<int,int*> maxPathSumHelper(const TreeNode* node, int* res)
{
    // base case
    if(node == nullptr)
        return {0, res};

    // get left side sum
    int left = maxPathSumHelper(node->left,res).first;

    // get right side sum
    int right = maxPathSumHelper(node->right,res).first;

    // update res
    std::vector<int> res_values = {*res,
                                   left+node->val, right+node->val,
                                   left+right+node->val, node->val};
    *res = *std::max_element(res_values.begin(),res_values.end());

    // path values
    std::vector<int> path_values = {left+node->val,right+node->val,node->val};
    return {*std::max_element(path_values.begin(),path_values.end()), res};

}

int maxPathSum(const TreeNode* node)
{
    int max_sum = std::numeric_limits<int>::min();
    int* max_sum_ptr = &max_sum;
    maxPathSumHelper(node, max_sum_ptr);
    return *max_sum_ptr;
}

int main()
{
    TreeNode* root = new TreeNode(10);

    root->left = new TreeNode(2);
    root->right = new TreeNode(10);

    root->left->left = new TreeNode(20);
    root->left->right = new TreeNode(1);

    root->right->right = new TreeNode(-25);
    root->right->right->left = new TreeNode(3);
    root->right->right->right = new TreeNode(4);

    std::cout << "Max Path Sum is: " << maxPathSum(root) << ".\n";

    return 0;
}

//        10
//       /  \
//      2    10
//     / \     \
//    20  1   -25
//             / \
//            3   4

