# Problem: Validate Binary Search Tree

# Description: Given a binary tree, determine if it is a valid binary search tree (BST).

# A valid BST is defined as follows:

# The left subtree of a node contains only nodes with keys less than the node's key.
# The right subtree of a node contains only nodes with keys greater than the node's key.
# Both the left and right subtrees must also be valid BSTs.

#      2
#    /   \
#   1     3

from typing import Optional

class TreeNode:
    def __init__(self,val: int, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val: int = val
        self.left: Optional['TreeNode'] = left
        self.right: Optional['TreeNode'] = right




def isValidBST(node: TreeNode) -> bool:
    # base case
    if node is None:
        return True
    
    # check none
    if node.right is None and node.left is None:
        return True
    
    # check left and right
    is_left = isValidBST(node.left)
    is_right = isValidBST(node.right)
    
    # do checks if left and right are valid
    if is_left and is_right:
        # if all nodes aren't none
        if ((node.left and node.val > node.left.val) and (node.right and node.val < node.right.val)):
            return True
        
        # only check right
        if node.left is None and node.right:
            if node.val < node.right.val:
                return True
            
        # only check left
        if node.right is None and node.left:
            if node.val > node.left.val:
                return True
            

    
    # all checks failed so return false
    return False


root = TreeNode(2)
root.left = TreeNode(1)
root.right = TreeNode(3)

print(f"Is valid BST? {isValidBST(root)}")

root = TreeNode(3)
root.left = TreeNode(2)
root.left.left = TreeNode(1)
root.right = TreeNode(5)
root.right.left = TreeNode(4)

print(f"Is valid BST? {isValidBST(root)}")