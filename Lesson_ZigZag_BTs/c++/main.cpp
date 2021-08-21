#include <stack>
#include <string>
#include <vector>
#include <iostream>



class Node
{
    public:
        int val;
        Node* left;
        Node* right;

        Node(int value, Node* left=nullptr, Node* right=nullptr)
        {
            this->val = value;
            this->left = left;
            this->right = right;
        }
};


std::string zz(Node* & root)
{
    // initialize data structures
    std::stack<Node*>* C = new std::stack<Node*>;
    std::stack<Node*>* N = new std::stack<Node*>;

    C->push(root);

    std::string res = "";
    int level = 0;

 
    // main process loop 
    while(C->size()>0)
    {
        // process C
        while(!C->empty())
        {
            Node* node = C->top();
            if(level % 2 == 0)
            {
                // go left, push children onto N
                if(node->left!=nullptr)
                    N->push(node->left);
                if(node->right!=nullptr)
                    N->push(node->right);
            }
            else
            {
                // go right, push children onto N
                if(node->right!=nullptr)
                    N->push(node->right);
                if(node->left!=nullptr)
                    N->push(node->left);
            }

            // add to result
            res += std::to_string(node->val);

            // remove from C
            C->pop();
        }

        // set C = N
        C = N;
        // empty N
        N = new std::stack<Node*>;
        //update level
        level += 1;

    }
    return res;
}



int main()
{

    Node* n7 = new Node(7);
    Node* n6 = new Node(6);
    Node* n5 = new Node(5);
    Node* n4 = new Node(4);
    Node* n3 = new Node(3, n6, n7);
    Node* n2 = new Node(2, n4, n5);
    Node* n1 = new Node(1, n2, n3);

    std::cout << zz(n1) << "\n";    

    return 0;
}