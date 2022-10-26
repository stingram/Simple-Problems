# A unival tree (which stands for "universal value") is a tree where all nodes under it have the same value.
# Given the root to a binary tree, count the number of unival subtrees.
# For example, the following tree has 5 unival subtrees:

#    0
#   / \
#  1   0
#     / \
#    1   0
#   / \
#  1   1


class Node(object):
    def __init__(self, val: int, left: 'Node' = None, right: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right


def unival(node: Node) -> int:
    def unival_helper(node: Node):
        if node is None:
            return (True, 0)
        res  = 0
        
        is_unival_right, num_right = unival_helper(node.right)
        is_unival_left, num_left = unival_helper(node.left)
        
        if node.left and node.right:
            if node.val == node.right.val and node.val == node.left.val and is_unival_right and is_unival_left:
                res = 1 + num_right + num_left
                return (True, res)

        elif node.left and not node.right:
            if node.val == node.left.val and is_unival_left:
                res = 1 + num_left
                return (True, res)

        elif node.right and not node.left:
            if node.val == node.right.val and is_unival_right:
                res = 1 + num_right
                return (True, res)

        else:
            return (True, 1)
        
        return (False, num_right + num_left)
        
        
    return unival_helper(node)[1]


root = Node(0)

root.left = Node(1)
root.right = Node(0)

root.right.left = Node(1)
root.right.right = Node(0)

root.right.left.left = Node(1)
root.right.left.right = Node(1)

print(unival(root))