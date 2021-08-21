#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

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


int is_balanced_helper(Node* & node)
{
    if(node==nullptr)
        return 0;

    int left_level;
    int right_level;

    left_level = is_balanced_helper(node->left);
    right_level = is_balanced_helper(node->right);

    if(left_level>=0 && right_level>=0 && std::abs(left_level-right_level) <= 1)
    {
        return std::max(left_level, right_level) + 1;
    }
    return -1;

}


bool is_balanced(Node* & node)
{
    return is_balanced_helper(node) != -1;
    
}

int main()
{
Node * n4 = new Node(4);
Node* n3 = new Node(3);
Node* n2 = new Node(2, n4);
Node* n1 = new Node(1, n2, n3);

//      1
//    / \
//   2   3
//  /
// 4
std::cout << std::boolalpha << is_balanced(n1) << "\n";

// True

n4 = new Node(4);
n2 = new Node(2, n4);
n1 = new Node(1, n2, nullptr);

//      1
//     /
//    2
//   /
//  4
std::cout << is_balanced(n1) << "\n";

// False

    
    return 0;
}