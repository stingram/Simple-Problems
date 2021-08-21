#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>

class Node
{
    private:
        int val = 0;
        Node* next = nullptr;
    public:
    Node(int val){
        this->val = val;
    }
    Node* get_next()
    {
        return next;
    }
    int get_value()
    {
        return val;
    }

    void set_next(Node* in_node)
    {
        next=in_node;
    }
};

Node* remove_kth_elem_from_end(Node* head, int k)
{
    // Need two pointers
    Node* slow = head;
    Node* fast = head;

    // Advance fast pointer k times
    int i = 0;
    while(fast && i < k)
    {
        fast = fast->get_next();
        i++;
        std::cout << "FAST: " << fast->get_value() << "\n";
    }
    // if we got to end early, remove head by returning list
    // starting with 2nd element
    if (fast == nullptr)
        return head->get_next();


    Node* prev = NULL;
    // go all the way to the end
    while(fast){
        // set previous to slow
        prev = slow;
        //avdance pointers
        fast = fast->get_next();
        slow = slow->get_next();
    }
    // skip over kth from last element
    prev->set_next(slow->get_next()); 
    delete slow;
    return head;
}


int main()
{
    Node* head = new Node(5);
    
    head->set_next(new Node(4));
    head->get_next()->set_next(new Node(3));
    head->get_next()->get_next()->set_next(new Node(2));
    head->get_next()->get_next()->get_next()->set_next(new Node(1));

    Node* res = remove_kth_elem_from_end(head, 3);


    while(res)
    {
        std::cout << "Val: " << res->get_value() << "\n";
        res=res->get_next();
    }

    return 0;
}