#include <vector>
#include  <iostream>


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

TreeNode* invert_tree(TreeNode* node)
{
    // base case
    if(node == nullptr)
        return node;
    
    // invert left
    TreeNode* inverted_left = invert_tree(node->left);

    // invert right
    TreeNode* inverted_right = invert_tree(node->right);

    // swap left and right
    node->left = inverted_right;
    node->right = inverted_left;

    // return node
    return node;
}

int main()
{
    TreeNode* root = new TreeNode(4);
    root->left = new TreeNode(2);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(3);

    root->right = new TreeNode(7);
    root->right->left = new TreeNode(6);
    root->right->right = new TreeNode(9);

    invert_tree(root);


    return 0;
}
