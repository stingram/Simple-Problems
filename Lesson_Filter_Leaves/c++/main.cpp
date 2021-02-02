#include <iostream>
#include <string>



class Node
{
    public:
    int val;
    Node* left;
    Node* right;
    Node(int val, Node* left=nullptr, Node* right=nullptr)
    {
        this->val = val;
        this->left = left;
        this->right = right;
    }

};


Node* filter_leaves(Node*& node, int keep_val)
{
    if(node == nullptr)
    {
        return nullptr;
    }

    node->left = filter_leaves(node->left, keep_val);
    node->right = filter_leaves(node->right, keep_val);

    if(node->val != keep_val && node->left==nullptr && node->right==nullptr)
    {
        return nullptr;
    }

    return node;

}

std::string print_tree(Node*& node)
{
    if(node!=nullptr)
    {
        return  std::to_string(node->val) + ",(" + print_tree(node->left) + ", " + print_tree(node->right) + ")";
    }
    else{
        return "";
    }
    
}


int main()
{
    Node* n2 = new Node(1);
    n2->left = new Node(2);
    n2->right = new Node(1);
    n2->left->left = new Node(2);
    n2->right->left = new Node(1);
    Node* n3 = filter_leaves(n2,2);
    std::cout << print_tree(n3) << "\n";
  
    return 0;
}



//     1
//    / \
//   2   1
//  /   /
// 2   1