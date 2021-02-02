class Node:
    def __init__(self, value, left=None, right=None):
        self.val = value
        self.left = left
        self.right = right

    def __repr__(self):
            return f"{self.val}, ({self.left.__repr__()}), ({self.right.__repr__()})"


def remove_leaves_helper(node: Node, filter_val: int):
    if node is None:
        return 1
    
    l = remove_leaves_helper(node.left, filter_val)
    r = remove_leaves_helper(node.right, filter_val)
    
    if(l == 1):
        node.left = None
    if(r == 1):
        node.right = None

    # print(f"Node val : {node.val}")
    
    # even after removing underylying leaves, this node still has children, don't remove it
    if(node.left is not None or node.right is not None):
        # print(f"Node has children: {node.left.val}, {node.right.val}")
        return 0
    
    # has no children, only remove if the value needs to be filtered
    else:
        if(node.val == filter_val):
            return 1
    
    # don't remove this node
    return 0

def remove_leaves(node: Node, filter_val: int):
    head = node
    remove_leaves_helper(node, filter_val)
    return head


def filter_leaves(node: Node, keep_val: int):
    if node is None:
        return None

    # filter left and right
    node.left = keep_leaves(node.left,keep_val)
    node.right = keep_leaves(node.right,keep_val)

    # remove if it's not value we want to keep
    if(node.val != keep_val and node.left is None and node.right is None):
        return None

    return node


#     1
#    / \
#   2   1
#  /   /
# 2   1
# n1 

n1 = Node(1, Node(2, Node(2), Node(1, Node(1))))
print(remove_leaves(n1, 2))

n2 = Node(1, Node(2, Node(2), Node(1, Node(1))))
print(keep_leaves(n2,2))

n1 = Node(1)
n1.left = Node(2)
n1.right = Node(1)
n1.left.left = Node(2)
n1.right.left = Node(1)
print(remove_leaves(n1, 2))

n2 = Node(1)
n2.left = Node(2)
n2.right = Node(1)
n2.left.left = Node(2)
n2.right.left = Node(1)
print(keep_leaves(n2,2))
