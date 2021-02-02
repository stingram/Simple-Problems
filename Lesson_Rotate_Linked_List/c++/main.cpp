#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>



class Node
{
    public:
    int val;
    Node* next;
    Node(int val, Node* next=NULL)
    {
        this->val=val;
        this->next = next;
    }
};

Node* rotate_list(Node* & node, int n)
{
    // get length of list
    int length = 0;
    Node* head = node;
    Node* curr = node;
    while(curr)
    {
        curr = curr->next;
        length+=1;
    }

    if(length == 0 or length == 1 or n == 0)
    {
        return node;
    }

    // set rotation
    n = n % length;

    // make fast and slow pointers
    Node* fast = node;
    Node* slow = node;

    // Advance fast pointer to spot n
    for(int i=0;i<n;i++)
    {
        fast = fast->next;
    }

    // advance both pointers until fast reaches end
    while(fast->next != NULL)
    {
        slow = slow->next;
        fast = fast->next;
    }

    // now set pointers appropriately
    fast->next = head;
    head = slow->next;
    slow->next = NULL;

    return head;

}


int main()
{
    Node * node = new Node(1, new Node(2, new Node(3, new Node(4, new Node(5)))));

    Node* node2 = rotate_list(node, 2);

    std::cout << node2->val << "\n";
    return 0;
}