# Description: Given a binary tree, flatten it to a linked list in-place.
# The linked list should be in the same order as a pre-order traversal of the binary tree.

#      1
#     / \
#    2   5
#   / \   \
#  3   4   6

# 1
#  \
#   2
#    \
#     3
#      \
#       4
#        \
#         5
#          \
#           6

from typing import Tuple

class TreeNode:
    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right

# return format is (head,tail)
def flattenHelper(node: TreeNode) -> Tuple[TreeNode,TreeNode]:
    # base case node is none:
    if node is None:
        return (None,None)
    
    # if node has node children
    if node.left is None and node.right is None:
        return (node,node)
    
    # flatten children
    left_head, left_tail = flattenHelper(node.left)
    right_head, right_tail = flattenHelper(node.right)
    
    # rearrange
    node.right = left_head
    
    # set left tail to point to right head
    if left_tail is not None:
        left_tail.right = right_head
    # if there was no left side at all then this node.right needs to point to right head
    else:
        node.right = right_head
    node.left = None
    
    return (node,right_tail)


def flatten(node: TreeNode) -> None:
    flattenHelper(node)
    return



#      1
#     / \
#    2   5
#   / \   \
#  3   4   6
root = TreeNode(1)
root.left = TreeNode(2)
root.left.left = TreeNode(3)
root.left.right = TreeNode(4)
root.right = TreeNode(5)
root.right.right = TreeNode(6)

flatten(root)

node = root
while node:
    print(f"{node.val}")
    node = node.right
    
    
