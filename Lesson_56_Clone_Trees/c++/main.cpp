#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>


class Node
{
    public:
    int val;
    Node* left;
    Node* right;

    Node(int val)
    {
        this->val = val;
        this->left = nullptr;
        this->right = nullptr;
    }
};

Node* find_node(Node* a, Node* b, Node* node)
{
    if(a==node)
    {
        return b;
    }
    Node* found;
    if(a->left != nullptr && b->left != nullptr)
    {
        found = find_node(a->left, b->left, node);
        if(found != nullptr)
        {
            return found;
        }
    }
    if(a->right != nullptr && b->right != nullptr)
    {
        found = find_node(a->right, b->right, node);
        if(found != nullptr)
        {
            return found;
        }
    }
    return nullptr;
}

int main()
{
    Node* a = new Node(1);
    a->left = new Node(2);
    a->right = new Node(3);
    a->right->left = new Node(4);
    a->right->right = new Node(5);

    Node* b = new Node(1);
    b->left = new Node(2);
    b->right = new Node(3);
    b->right->left = new Node(4);
    b->right->right = new Node(5);

    Node* found = find_node(a,b,a->right->left);
    std::cout << found->val << "\n";

    return 0;
}

// class Node:
//     def __init__(self, val):
//         self.val = val
//         self.left = None
//         self.right = None
        
//     def __str__(self):
//         return str(self.val)
        
        
// def find_node(a: Node, b: Node, node: Node) -> Node:
//     if a == node:
//         return b
//     if a.left and b.left:
//         found = find_node(a.left, b.left, node)
//         if found:
//             return found
//     if a.right and b.right:
//         found = find_node(a.right, b.right, node)
//         if found:
//             return found
//     return None


// #  1
// # / \
// #2   3
// #   / \
// #  4*  5
// a = Node(1)
// a.left = Node(2)
// a.right = Node(3)
// a.right.left = Node(4)
// a.right.right = Node(5)

// b = Node(1)
// b.left = Node(2)
// b.right = Node(3)
// b.right.left = Node(4)
// b.right.right = Node(5)

// print(find_node(a, b, a.right.left))
// # 4