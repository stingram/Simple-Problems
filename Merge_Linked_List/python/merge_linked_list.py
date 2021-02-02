class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next= next

    def __repr__(self):
        return f"{self.val}, {self.next}"

def merge_list(l1, l2):
    c1 = l1
    c2 = l2

    l3 = Node(0)
    c3 = l3

    while(c1 and c2):
        if c1.val < c2.val:
            c3.next = c1
            c1 = c1.next
        else:
            c3.next = c2
            c2 = c2.next
        c3 = c3.next
    while(c1):
        c3.next = c1
        c1 = c1.next
        c3 = c3.next
    while(c2):
        c3.next = c2
        c2 = c2.next
        c3 = c3.next

    return l3.next

l1 = Node(0, Node(2, Node(4, Node(6))))
l2 = Node(1, Node(3, Node(5, Node(7))))

print(merge_list(l2,l1))

l1 = Node(1, Node(3, Node(5, Node(7))))
l2 = Node(1, Node(3, Node(5, Node(7))))

print(merge_list(l2,l1))