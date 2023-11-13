# Description: Given a binary tree, you need to compute the length of the diameter of the tree.
# The diameter of a binary tree is the length of the longest path between any two nodes in a tree.
# This path may or may not pass through the root.

#      1
#     / \
#    2   3
#   / \
#  4   5
# Ans: 4

from typing import Tuple

class TreeNode:
    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right

def diameterOfBinaryTreeHelper(node: TreeNode,longest: int) -> Tuple[int,int]:
    # base case
    if node is None:
        return (0, longest)
    
    # if no children
    if node.left is None and node.right is None:
        if(1 > longest):
            longest = 1
        return (1, longest)
    
    # get longest from left branch
    left_side, _ = diameterOfBinaryTreeHelper(node.left, longest)
    
    # get longest from right branch
    right_side, _ = diameterOfBinaryTreeHelper(node.right, longest)
    
    # update longest if left_side+right_side+1 is longest
    longest = max(longest, left_side+right_side+1)
    
    # we can choose one side for the path 
    return (max(left_side,right_side)+1,longest)

def diameterOfBinaryTree(root: TreeNode) -> int:
    return diameterOfBinaryTreeHelper(root,0)[1]


root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.left.right.right = TreeNode(6)
print(f"Diameter of tree is: {diameterOfBinaryTree(root)}.")