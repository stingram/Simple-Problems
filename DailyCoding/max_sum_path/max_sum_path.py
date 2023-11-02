

# Given a binary tree, find the maximum path sum. 
# The path may start and end at any node in the tree, 
# but you must traverse the parent-child connections in the tree

#        10
#       /  \
#      2    10
#     / \     \
#    20  1   -25
#             / \
#            3   4

from typing import Tuple

class TreeNode:
    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val: int = val
        self.left: TreeNode = left
        self.right: TreeNode = right

def maxPathSumHelper(node: TreeNode, res: int) -> Tuple[int,int]:
    # Base case
    if node is None:
        return 0, res
    
    # get max from left
    left = maxPathSumHelper(node.left, res)[0]
    
    # get max from right
    right = maxPathSumHelper(node.right, res)[0]
    
    # Update res with max of choice between left path, right path, splitting here, just taking this node or taking none of these node
    # res = max(0, res, left + this node, right + this node, left + this node + right, this node)
    res = max(0, res, left+node.val, right+node.val, left+right+node.val, node.val)
    
    # only return
    # max(left + this node, right + this node, node, 0)
    return max(left+node.val, right+node.val, node.val, 0), res

def maxPathSum(node:TreeNode) -> int:
    maxSum = float('-inf')
    return maxPathSumHelper(node, maxSum)[1]



#        10
#       /  \
#      2    10
#     / \     \
#    20  1   -25
#             / \
#            3   4
root = TreeNode(10)
root.left = TreeNode(2)
root.left.left = TreeNode(20)
root.left.right = TreeNode(1)

root.right = TreeNode(10)
root.right.right = TreeNode(-25)
root.right.right.left = TreeNode(3)
root.right.right.right = TreeNode(4)



print(f"Max sum for graph is: {maxPathSum(root)}.")

