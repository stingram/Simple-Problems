#include <cstdlib>
#include <iostream>


struct Node {
    int data;
    struct Node* next;
};

struct Node* new_node(int key)
{
    struct Node* temp = new Node;
    temp->data = key;
    temp->next = NULL;
    return temp;
}

void printlist(struct Node* node)
{
    while(node!=NULL)
    {
        std::cout << " "<< node->data;
        node = node->next;
    }
}

// Merge two lsits with headers as h1 and h2
// It assumes that h1's data is smaller than
// or equal to h2's data.
struct Node* merge_util(struct Node* h1,
                        struct Node* h2)
{
    // if only one node in first list
    // simply point its head to second list
    if(!h1->next)
    {
        h1->next = h2;
    }
    // Initialize current and next pointers of
    // both lists
    struct Node* curr1 = h1, *next1 = h1->next;
    struct Node* curr2 = h2, *next2 = h2->next;

    // while next1 exists and curr2 exists
    while(next1 && curr2)
    {
        // if curr2 lies in between curr1 and next1
        // then do cur1->curr2->next1
        if((curr2->data) >= (curr1->data) && (curr2->data) <=(next1->data))
        {
            // Make sure next2 is correct before changing curr2
            next2 = curr2->next;
            curr1->next = curr2;
            curr2->next = next1;

            // now let curr1 and curr2 point
            // to their immediate next pointers
            // basically, advance each appropriately
            // curr1 next is now curr2
            curr1 = curr2;
            // curr2 next is old curr2's next
            curr2 = next2;
        }
        // if got here, then curr2 isn't between curr1 and next1
        // so we need to update curr1 and next1
        else
        {
            // if more nodes in first list
            // aka, if it's possible to advance next1 and curr1
            if(next1->next)
            {
                next1 = next1->next;
                curr1 = curr1->next;
            }
            // else, we've reached the end of our first list so we
            // point the last node of first list
            // to the remaining nodes of second list
            else{
                next1->next=curr2;
                return h1;
            }
        }
        
    }
    return h1;
}

// Merges two given lists in-place. This function
// mainly compares head nodes and calls mergeUtil()
struct Node* merge(struct Node* h1,
                    struct Node* h2)
{
    // edge cases
    if(!h1)
        return h2;
    if(!h2)
        return h1;

    // start with linked list
    // whose head data is the least
    if(h1->data < h2->data)
        return merge_util(h1, h2);
    else
        return merge_util(h2, h1);
}


// Driver program
int main()
{
    struct Node* head1 = new_node(1);
    head1->next = new_node(3);
    head1->next->next = new_node(5);

    struct Node* head2 = new_node(0);
    head2->next = new_node(2);
    head2->next->next = new_node(4);

    struct Node* merged_head = merge(head1, head2);

    printlist(merged_head);
    return 0;

}