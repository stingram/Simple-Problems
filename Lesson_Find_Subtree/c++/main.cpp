#include <iostream>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>


class Node{
    public:

    int value;
    Node* left;
    Node* right;

    Node(int value)
    {
        this->value=value;
        this->left=NULL;
        this->right=NULL;
    }
};

std::string serialize(Node* a)
{
    if(a==NULL)
    {
        return "x";
    }
    return std::to_string(a->value) + serialize(a->left) + serialize(a->right);
}


bool find_subtree(Node* a, Node* b)
{
    // serialize a
    std::string serial_a = serialize(a);

    // serialize b
    std::string serial_b = serialize(b);

    // find if string b in string a
    if(serial_a.find(serial_b) != std::string::npos)
    {
        return true;
    }

    return false;
}


int main()
{
    Node* a=new Node(1);
    a->left = new Node(2);
    a->right = new Node(3);
    a->left->left = new Node(4);
    a->left->right = new Node(5);
    a->right->left = new Node(6);
    a->right->right = new Node(7);

    Node* b = new Node(3);
    b->left = new Node(6);
    b->right = new Node(7);

    std::cout << "B is subtree of A: " << find_subtree(a,b) << ".\n";

    return 0;
}