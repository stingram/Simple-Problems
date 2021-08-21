#include <cstdlib>
#include <iostream>
#include <string>



class Node
{
    private:
    int val;
    Node* left;
    Node* right;

    public:
    Node(int val)
    {
        this->val = val;
        this->left = nullptr;
        this->right = nullptr;
    }
    void set_left(Node* left){
        this->left = left;
    }
    void set_right(Node* right){
        this->right = right;
    }

    Node* get_left(){
        return this->left;
    }
    Node* get_right(){
        return this->right;
    }

};

class Solution
{
    Node* invert(Node* node)
    {
        if(node == nullptr){
            return nullptr;
        }
        node->set_left(invert(node->get_left()));
        node->set_right(invert(node->get_right()));
        Node* temp = node->get_left();
        node->set_left(node->get_right());
        node->set_right(temp);
        return node;
    }
};

int main()
{
    Node* root = new Node(1);
    root->set_left(new Node(2));
    root->set_right(new Node(3));
    return 0;
}

// class Node:
//     def __init__(self, val):
//         self.val = val
//         self.left = None
//         self.right = None
        
// class Solution:
//     def invert(self, node: Node) -> Node:
//         if node is None:
//             return
//         node.left = self.invert(node.left)
//         node.right = self.invert(node.right)
        
//         temp = node.left
//         node.left = node.right
//         node.right = temp
//         return node