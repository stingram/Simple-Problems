#include <vector>
#include <iostream>


class Node
{
    public:
        int val;
        Node* next;
        Node(int value)
        {
            val = value;
            next = nullptr;
        }
};


void swap_values(Node*& node)
{
    Node* curr = node;
    int temp;
    while(curr && curr->next)
    {
        temp = curr->val;
        curr->val = curr->next->val;
        curr->next->val = temp;

        // advance pointer
        curr = curr->next->next;
    }
}


int main()
{
    Node* my_list = new Node(1);
    my_list->next = new Node(2);
    my_list->next->next = new Node(3);
    my_list->next->next->next = new Node(4);
    my_list->next->next->next->next = new Node(5);
    my_list->next->next->next->next->next = new Node(6);
    // my_list->next->next->next->next->next->next = new Node(7);

    swap_values(my_list);
    
    Node* c = my_list;
    while(c)
    {
        std::cout << c->val << " ";
        c = c->next;
    }
    std::cout << "\n";
    return 0;
}
