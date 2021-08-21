# Definition for a binary tree node.
class TreeNode:
  def __init__(self, x):
    self.val = x
    self.left = None
    self.right = None

class Solution(object):
    def _height(self, node: TreeNode, height: int) -> int:
        if node:
            return 1 + max(self._height(node.left, height), self._height(node.right, height))
        return height
    
    def is_balanced(self, node: TreeNode, height=0) -> bool:
        if node:
            h1 = self._height(node.left, height+1)
            h2 = self._height(node.right, height+1)
            if abs(h1-h2) <= 1 and self.is_balanced(node.left) and self.is_balanced(node.right):
                return True
            return False
        return True
        
        
        def is_balanced_v2(self, root: TreeNode) -> bool:
            def is_balanced_helper(root):
                # base case
                if root == None:
                    return (True, 0)
                
                # get heights and balance of left and right subtrees
                leftB, leftH = is_balanced_helper(root.left)
                rightB, rightH = is_balanced_helper(root.right)
                
                # return balance and height
                return (leftB and rightB and abs(leftH-rightH) <= 1, max(leftH, rightH) + 1)
            return is_balanced_helper(root)[0]

    
    
tree = TreeNode(5)
tree.left = TreeNode(4)
tree.right = TreeNode(3)
tree.left.left = TreeNode(6)
tree.left.right = TreeNode(7)
tree.left.left.left = TreeNode(8)
tree.left.left.right = TreeNode(9)
tree.left.right.left = TreeNode(10)
tree.left.right.right = TreeNode(11)

tree.right.left = TreeNode(12)
tree.right.right = TreeNode(13)

print(Solution().is_balanced(tree))
