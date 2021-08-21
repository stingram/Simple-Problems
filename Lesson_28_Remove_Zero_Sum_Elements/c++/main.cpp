#include <cstdlib>
#include <list>
#include <iostream>
#include <string>
#include <map>
#include <iterator>


// Definition for singly-linked list.
class ListNode{
  
    public:
    int val;
    ListNode* next;

    ListNode(int input)
    {
        val = input;
        next = nullptr;
    }
};
    
class Solution{
    public:
    ListNode* remove_zero_sum(ListNode* head){
        // set curr node and dummy prefix sum node to lead head node
        ListNode* curr = new ListNode(0);
        ListNode* dummy = curr;
        dummy->next = head;
        // set prefix value
        int prefix = 0;
        
        // Ordered Hash Map to keep track of prefix values we've seen
        // mapped to the node where it occurred 
        std::map<int, ListNode*> seen;
        
        // process list
        while(curr != nullptr){
            // add current value to prefix sum
            prefix += curr->val;
            
            // if the prefix value hasn't been seen, just add to hash map
            if(seen.find(prefix) == seen.end())
                seen.insert(std::pair<int,ListNode*>(prefix, curr));
                
            // if we have seen this prefix value before, we need to remove
            // curr node and all nodes after the node at original prefix 
            else{
                // get original node
                ListNode* node = seen[prefix];
                // use original to skip over zero sum nodes
                node->next = curr->next;
                // Now we need to remove those prefix keys for those
                // removed nodes
                // So we get the list of keys in order, and we remove
                // key,value pairs until we get to original prefix
                std::map<int,ListNode*>::iterator it = seen.find(prefix);
                seen.erase(it, seen.end());    
            }
            // advance current node
            curr = curr->next;
        }
        // return original head
        return dummy->next;
    }
};

int main()
{
    ListNode* n1 = new ListNode(0);
    ListNode* n2 = new ListNode(2);
    n1->next = n2;
    
    return 0;
}
