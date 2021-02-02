class Node():
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        


def in_order_serialize(g):
    if g is None:
        return 'x'
    return str(g.value) + in_order_serialize(g.left) + in_order_serialize(g.right)

def find_subtree(n, b):
    
    # serialize n
    n_serial = in_order_serialize(n)
    
    # serialize b
    b_serial = in_order_serialize(b)
    
    # 
    print(n_serial)
    
    if b_serial in n_serial:
        return True
    
    return False


def find_subtree2(n, b):
    if not n and not b:
        return True
    if not n:
        return False
    
    if not b:
        return True
    
    is_match = n.value == b.value
    
    if is_match:
        is_match_left = find_subtree2(n.left,b.left)
        
        if is_match_left:
            is_match_right = find_subtree2(n.right,b.right)
            
            if is_match_right:
                return True
            
    # check rest of n    
    return find_subtree2(n.left,b) or find_subtree2(n.right,b)


n = Node(1)
n.left = Node(4)
n.right = Node(5)
n.left.left = Node(3)
n.left.right = Node(2)
n.right.left = Node(4)
n.right.right = Node(1)


b = Node(4)
b.left = Node(3)
b.right = Node(2)

print(find_subtree(n,b))

print(find_subtree2(n,b))