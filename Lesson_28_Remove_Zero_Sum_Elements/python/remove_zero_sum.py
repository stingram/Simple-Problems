import collections

# Definition for singly-linked list.
class ListNode:
  def __init__(self, x):
    self.val = x
    self.next = None
    
    
class Solution:
    def remove_zsero_sum(self, head):
        # set curr node and dummy prefix sum node to lead head node
        curr =dummy = ListNode(0)
        dummy.next = head
        # set prefix value
        prefix = 0
        
        # Ordered Hash Map to keep track of prefix values we've seen
        # mapped to the node where it occurred 
        seen = collections.OrderedDict()
        
        # process list
        while curr:
            # add current value to prefix sum
            prefix += curr.val
            
            # if the prefix value hasn't been seen, just add to hash map
            if prefix not in seen:
                seen[prefix] = curr
                
            # if we have seen this prefix value before, we need to remove
            # curr node and all nodes after the node at original prefix 
            else:
                # get original node
                node = seen[prefix]
                # use original to skip over zero sum nodes
                node.next = curr.next
                # Now we need to remove those prefix keys for those
                # removed nodes
                # So we get the list of keys in order, and we remove
                # key,value pairs until we get to original prefix
                while list(seen.keys())[-1] != prefix:
                    seen.popitem()
            
            # advance current node
            curr = curr.next
        
        # return original head
        return dummy.next