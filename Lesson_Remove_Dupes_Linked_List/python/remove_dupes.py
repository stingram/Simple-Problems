class Node(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
        
    def __repr__(self):
        return str(self.val) + ", " + str(self.next)
        

def remove_dupes(node: Node):
    
    # build dictionary of counts of elements
    curr = node
    vals_dict = {}
    while curr:
        if curr.val not in vals_dict:
            vals_dict[curr.val] = 1
            
        else:
            vals_dict[curr.val] += 1
        curr = curr.next
    
    # Go through list again removing elements that are in dictionary where value is > 2
    curr = node
    prev = None
    while curr:
        if vals_dict[curr.val] > 1:
            
            # remove from dict
            vals_dict[curr.val] -= 1
            
            # remove this element
            
            # takes care of needing to remove first element
            if prev:
                prev.next = curr.next
            curr = curr.next

        else:
            prev = curr
            curr = curr.next
            
def remove_dupes_in_order(node: Node):
    
    curr = node
    while curr and curr.next:
        if curr.val == curr.next.val:
            curr.next = curr.next.next
        else:
            curr = curr.next
            
            
node = Node(1, Node(2, Node(2, Node(3, Node(3)))))
remove_dupes_in_order(node)
print(node)