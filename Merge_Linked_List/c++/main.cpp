#include <iostream>
#include <vector>
#include <string>

class Node
{
    public:
    int val;
    Node* next;
        Node(int val, Node* next = nullptr)
        {
            this->val=val;
            this->next=next;
        }
};

Node* merge_lists(Node* & l1, Node* & l2)
{
    Node* c1 = l1;
    Node* c2 = l2;
    Node* l3 = new Node(0);
    Node* c3 = l3;

    while(c1 != nullptr && c2 != nullptr)
    {
        if(c1->val < c2->val)
        {
            c3->next = c1;
            c1=c1->next;
        }
        else
        {
            c3->next = c2;
            c2 = c2->next;
        }
        c3 = c3->next;
    }
    while(c1 != nullptr)
    {
        c3->next = c1;
        c1 = c1->next;
        c3 = c3->next;
    }
    while(c2 != nullptr)
    {
        c3->next = c2;
        c2 = c2->next;
        c3 = c3->next;
    }

    return l3->next;


    return l3->next;
}

void print_list(Node* & l1)
{
    while(l1)
    {
        std::cout << l1->val << ",";
        l1 = l1->next;
    }
    std::cout <<"\n";
    return;
}

int main()
{
    Node* n1 = new Node(1, new Node(3, new Node(5, new Node(7))));
    Node* n2 = new Node(2, new Node(4, new Node(6, new Node(8))));

    Node* n3 = merge_lists(n1,n2);

    print_list(n3);

    n1 = new Node(1, new Node(3, new Node(5, new Node(7))));
    n2 = new Node(1, new Node(3, new Node(5, new Node(7))));

    n3 = merge_lists(n1,n2);

    print_list(n3);


    return 0;
}