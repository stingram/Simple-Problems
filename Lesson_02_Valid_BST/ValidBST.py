# Definition for a binary tree node.
class TreeNode:
  def __init__(self, x):
    self.val = x
    self.left = None
    self.right = None
    
class Solution:
  def isValidBST(self, root: TreeNode):
    def helper(node: TreeNode, lower: float, upper: float):
          if node is None:
                return True
          val = node.val
          if val < lower or val > upper:
                return False
          if not helper(node.left, lower, val):
                return False
          if not helper(node.right, val, upper):
                return False
          return True
    return helper(root, float('-inf'), float('inf'))
  
tree = TreeNode(5)
tree.left = TreeNode(3)
tree.right = TreeNode(11)

print(Solution().isValidBST(tree))