#include <vector>
#include <iostream>

class TreeNode
{
    public:
    TreeNode(int _val)
    {
        val = _val;
        left = nullptr;
        right = nullptr;
    }

    int val;
    TreeNode* left;
    TreeNode* right;
};



TreeNode* LCA(TreeNode*curr, TreeNode* p, TreeNode* q)
{
    // base case, we reach leave or a target node
    if(curr == nullptr || curr->val == p->val || curr->val == q->val)
    {
        return curr;
    } 

    // Recurse on left and right side
    TreeNode* left = LCA(curr->left,p,q);
    TreeNode* right = LCA(curr->right,p,q);

    // if left and right found, then this node is the LCA
    if(left != nullptr && right != nullptr)
    {
        return curr;
    }

    // if only left found, propagate tha information up
    else if(left != nullptr)
    {
        return left;
    }
    // if only right found, propagate that information up
    else if(right != nullptr)
    {
        return right;
    }

}



int main()
{
    TreeNode* t = new TreeNode(3);
    t->left = new TreeNode(5);
    t->right = new TreeNode(1);
    t->left->left = new TreeNode(6);
    t->left->right = new TreeNode(2);

    t->right->left = new TreeNode(0);
    t->right->right = new TreeNode(8);

    t->left->right->left = new TreeNode(7);
    t->left->right->right = new TreeNode(4);

    std::cout << "LCA of nodes 4 and 5 is : " << LCA(t,t->left,t->left->right->right)->val << ".\n";

    return 0;
}

// LCA of nodes 4 and 5 is: {LCA(t,t.left,t.left.right.right).val}

// t = TreeNode(3)
// t.left = TreeNode(5)
// t.right = TreeNode(1)
// t.left.left = TreeNode(6)
// t.left.right = TreeNode(2)

// t.right.left = TreeNode(0)
// t.right.right = TreeNode(8)

// t.left.right.left = TreeNode(7)
// t.left.right.right = TreeNode(4)