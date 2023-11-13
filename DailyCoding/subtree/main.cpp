#include <vector>
#include <iostream>
#include <deque>

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

TreeNode* find_root(TreeNode* s, TreeNode* t)
{
    if(s == nullptr || t == nullptr)
        return nullptr;
    std::deque<TreeNode*> q = {s};
    while(q.size() > 0)
    {
        TreeNode* curr_node = q.front();
        q.pop_front();
        if(curr_node->val == t->val)
        {
            return curr_node;
        }
        // try for children
        if(curr_node->left != nullptr)
            q.push_back(curr_node->left);
        if(curr_node->right != nullptr)
            q.push_back(curr_node->right);
    }
    return nullptr;
}

bool check_subtree(TreeNode* s, TreeNode* t)
{
    // if either are none
    if(s == nullptr && t == nullptr)
        return true;
    if(s == nullptr && t != nullptr)
        return false;
    if(s != nullptr && t == nullptr)
        return false;
    if(s->val == t->val)
    {
        bool check_subtree_left = check_subtree(s->left, t->left);
        bool check_subtree_right = check_subtree(s->right, t->right);
        if(check_subtree_left && check_subtree_right)
        {
            return true;
        }
    }
    return false;
}

bool is_subtree(TreeNode* s, TreeNode* t)
{
    TreeNode* sub_root = find_root(s,t);
    if(sub_root == nullptr)
        return false;
    return check_subtree(sub_root, t);
}

int main()
{
    TreeNode* s = new TreeNode(3);
    s->left = new TreeNode(4);
    s->right = new TreeNode(5);
    s->left->left = new TreeNode(1);
    s->left->right = new TreeNode(2);

    TreeNode* t = new TreeNode(4);
    t->left = new TreeNode(1);
    t->right = new TreeNode(2);

    std::cout << "t is a subtree of s? " << std::boolalpha << is_subtree(s,t) << ".\n";
    return 0;
}
