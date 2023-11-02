#include <vector>
#include <iostream>


class TreeNode
{
    public:
    TreeNode(int _val, TreeNode* _left = nullptr, TreeNode* _right = nullptr)
    {
        val = _val;
        left = left;
        right = right;
    }
    int val;
    TreeNode* left;
    TreeNode* right;
};


bool isValidBST(TreeNode* node)
{
    if(node == nullptr)
        return true;
    if(node->left == nullptr && node->right == nullptr)
        return true;
    
    // check left and right
    bool left_is_bst = isValidBST(node->left);
    bool right_is_bst = isValidBST(node->right);

    // check respective values
    if(left_is_bst && right_is_bst)
    {
        // if both are actual nodes
        if(node->right != nullptr and node->left != nullptr)
        {
            if(node->left->val < node->val && node->right->val > node->val)
            {
                return true;
            }
        }

        // if only left is a node
        if(node->left != nullptr && node->right == nullptr)
        {
            if(node->left->val < node->val)
            {
                return true;
            }
        }

        // last case must be if only right is a node
        if(node->left == nullptr && node->right != nullptr)
        {
            if(node->right->val > node->val)
            {
                return true;
            }
        }
    }


    // all checks failed
    return false;
}

int main()
{
    TreeNode* root = new TreeNode(2);
    root->left = new TreeNode(1);
    root->right = new TreeNode(3);
    root->right->right = new TreeNode(-1);
    std::cout << "Is valid BST? " << std::boolalpha << isValidBST(root) << ".\n"; 
    return 0;
}