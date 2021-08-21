#include <iostream>
#include <string>
#include <sstream>

class Node
{
    public:
    int val;
    Node* next;
    Node(int val, Node* next = NULL)
    {
        this->val = val;
        this->next = next;
    }
};



Node* remove_dupes(Node* & node)
{
    Node * curr = node;
    while(curr != NULL && curr->next != NULL)
    {
        if(curr->val == curr->next->val)
        {
            curr->next = curr->next->next;
        }
        else{
            curr = curr->next;
        }
    }
    return node;
}

void print_list(Node* & node)
{
    Node* curr = node;
    std::string str_rep = ""; 
    while(curr){
        str_rep += std::to_string(curr->val) + ",";
        curr = curr->next;
    }
    std::cout << str_rep << "\n";
}


int main()
{
    Node * node = new Node(1, new Node(2, new Node(2, new Node(3, new Node(3)))));
    print_list(node);

    Node* no_dupes = remove_dupes(node);
    print_list(no_dupes);

    return 0;
}