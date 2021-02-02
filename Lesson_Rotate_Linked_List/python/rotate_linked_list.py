class Node(object):
    def __init__(self, val, next = None):
        self.val = val
        self.next = next
        
        
class LL(object):
    def __init__(self, head=None):
        self.head = head
    
    def rotate(self, M):
        
        # need cases where list is len
        assert M > 0
        
        # get last node in list
        curr = self.head
        last = curr
        L=0
        while(curr):
            last = curr
            curr=curr.next
            L+=1
            
        # if L is 0 or 1 just return same list
        if L == 0 or L == 1:
            return
        

        
        # now we have length
        if M >= L:
            r = M % L
        else:
            r = M
        
        
        print("r: " + str(r))
        
        # setup
        prev = None
        new_first=self.head
        c = 0
        
        # rotate
        while(c < r):
            prev = new_first
            new_first = new_first.next
            c+=1
        
        # set values
        prev.next = None
        last.next = self.head
        self.head = new_first
        
        
        
# use slow and fast pointer
def rotate(node, n):
    length = 0
    curr = node
    # copmute length of linked list
    while curr != None:        
        curr = curr.next
        length += 1
        
    # number of shifts
    n = n % length
    
    slow, fast = node, node
    # advance fast to rotation spot
    for i in range(n):
        fast = fast.next

    # advance fast to end with slow following in lock step
    while fast.next is not None:
        slow = slow.next
        fast = fast.next
        
    fast.next = node
    head = slow.next
    slow.next = None
    
    return head
        
        
a = Node(1)
a.next = Node(2)
a.next.next = Node(3)
a.next.next.next = Node(4)
a.next.next.next.next = Node(5)
a.next.next.next.next.next = Node(6)

ll = LL(a)
        
ll.rotate(3)
# 4
print(ll.head.val)



a = Node(1)
a.next = Node(2)
a.next.next = Node(3)
a.next.next.next = Node(4)
a.next.next.next.next = Node(5)
a.next.next.next.next.next = Node(6)
ll = LL(a)
ll.rotate(9)
# 4
print(ll.head.val)        