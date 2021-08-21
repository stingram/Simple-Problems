class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        
    def __str__(self):
        return str(self.val)
        
        
def find_node(a: Node, b: Node, node: Node) -> Node:
    if a == node:
        return b
    if a.left and b.left:
        found = find_node(a.left, b.left, node)
        if found:
            return found
    if a.right and b.right:
        found = find_node(a.right, b.right, node)
        if found:
            return found
    return None


#  1
# / \
#2   3
#   / \
#  4*  5
a = Node(1)
a.left = Node(2)
a.right = Node(3)
a.right.left = Node(4)
a.right.right = Node(5)

b = Node(1)
b.left = Node(2)
b.right = Node(3)
b.right.left = Node(4)
b.right.right = Node(5)

print(find_node(a, b, a.right.left))
# 4