# Description: Find the sum of all left leaves in a given binary tree.
# A left leaf is a leaf node that is a child of the left subtree.

#      3
#     / \
#    9  20
#       / \
#      15  7
# Ans: 9 + 15 = 24

from typing import Tuple

class TreeNode:
    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right
        
def sumOfLeftLeavesHelper(node: TreeNode, sum: int, could_be_left_leaf: bool) -> Tuple[int, bool]:
    # base case, node is none
    if node is None:
        return (sum,False)
    
    # if node has no children
    if node.left is None and node.right is None:
        if could_be_left_leaf:
            return (sum+node.val, could_be_left_leaf)
        else:
            return (sum, could_be_left_leaf)
        
    # call helper on children
    sum = sumOfLeftLeavesHelper(node.left,sum,True)[0]
    sum = sumOfLeftLeavesHelper(node.right,sum,False)[0]
    
    return (sum,False)
        
def sumOfLeftLeaves(root: TreeNode) -> int:
    return sumOfLeftLeavesHelper(root, 0, False)[0]

#      3
#     / \
#    9  20
#       / \
#      15  7
# Ans: 9 + 15 = 24

root = TreeNode(3)
root.left = TreeNode(9)
root.left.left = TreeNode(-16)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(24)

print(f"Sum of left leaves is: {sumOfLeftLeaves(root)}.")