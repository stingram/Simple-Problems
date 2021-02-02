class Node():
    def __init__(self, val):
        self.val = val
        self.next = None
        
    def __repr__(self):
        if self.next:
            return str(self.val) + " " + str(self.next)
        else:
             return str(self.val)

    
class myList():
    def __init__(self, node: Node):
        self.head = node
        
    def __repr__(self):
        lstr = ""
        curr=head
        while(curr is not None):
            lstr += " " + str(curr)
        return lstr
        
    def swap_element(self,):
        return
    
    
def swap_elements(node: Node):
    head = node
    curr = head
    prev = None
    
    while(curr and curr.next):
        
        # start of list
        if prev == None:
            # swap
            head = curr.next
            curr.next = curr.next.next
            head.next = curr
            
            # advance pointers
            prev = head.next
            curr = curr.next
            
        # not at beginning
        else:
            # swap
            prev.next = curr.next
            curr.next = curr.next.next
            prev.next.next = curr
            
            # advance pointers
            prev = curr
            curr = curr.next
    
    return head

def swap_values(node: Node):
    curr = node
    while(curr and curr.next):
        curr.val, curr.next.val = curr.next.val, curr.val
        curr = curr.next.next
    return node
    

    
my_list = Node(1)
my_list.next = Node(2)
my_list.next.next = Node(3)
my_list.next.next.next = Node(4)
my_list.next.next.next.next = Node(5)
my_list.next.next.next.next.next = Node(6)
my_list.next.next.next.next.next.next = Node(7)
print(my_list)
print(swap_elements(my_list))

my_list = Node(1)
my_list.next = Node(2)
my_list.next.next = Node(3)
my_list.next.next.next = Node(4)
my_list.next.next.next.next = Node(5)
my_list.next.next.next.next.next = Node(6)
my_list.next.next.next.next.next.next = Node(7)
print(my_list)
print(swap_values(my_list))