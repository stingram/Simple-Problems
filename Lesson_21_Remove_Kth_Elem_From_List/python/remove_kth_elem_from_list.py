class Node:
    def __init__(self, val, next):
        self.val = val
        self.next = next
        
    def __str__(self):
        n = self
        answer = ''
        while n:
            answer += str(n.val)
            n = n.next
            return answer
        
def remove_kth_from_linked_list(node, k):
    slow, fast = node, node
    for i in range(k):
        fast = fast.next
    if not fast:
        return node.next
    
    prev = None
    while fast:
        # Need to keep node before the one we are to remove
        prev = slow
        fast = fast.next
        slow = slow.next
    
    # point previous to one after slow
    prev.next = slow.next
    return node
    
head = Node(1, Node(2, Node(3, Node(4, Node(5)))))
print(head)

head = remove_kth_from_linked_list(head, 1)

print(head)     