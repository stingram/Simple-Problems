class Node:
    def __init__(self, value, left=None, right=None, parent=None):
        self.val = value
        self.left = left
        self.right = right
        self.parent = parent

    def __repr__(self):
        r = "{v}, {l}, {r}".format(v=self.val, l=self.left, r=self.right)
        return r


def find_inorder_successor(node: Node):
    # if node doesn't have right child
    if node.right is None:
        n2 = node.parent
        while n2:
            if n2.val > node.val:
                return n2
            n2 = n2.parent
        
        # it doesn't have a right child or a parent, so returns None
        return n2
    
    # if it does have a right child, go down left side of right subtree
    else:
        n2 = node.right
        while n2.left:
            n2 = n2.left
        return n2
        
tree = Node(4)
tree.left = Node(2)
tree.right = Node(8)
tree.left.parent = tree
tree.right.parent = tree
tree.left.left = Node(1)
tree.left.left.parent = tree.left
tree.right.right = Node(7)
tree.right.right.parent = tree.right
tree.right.left = Node(5)
tree.right.left.parent = tree.right
tree.right.left.right = Node(7)
tree.right.left.right.parent = tree.right.left
tree.right.right = Node(9)
tree.right.right.parent = tree.right
#     4
#    / \
#   2   8
#  /   / \
# 1   5   9
#      \
#       7

print(find_inorder_successor(tree.right))
# 9

print(find_inorder_successor(tree.left))
# 4

print(find_inorder_successor(tree.right.left.right))
# 8