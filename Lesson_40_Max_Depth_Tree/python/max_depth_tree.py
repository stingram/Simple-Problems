class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def find_depth(self, node: TreeNode) -> int:
        def helper(node, height):
            if node is None:
                return height
            hl = helper(node.left, height + 1)
            hr = helper(node.right, height + 1)
            return max(hl, hr)
        
        return helper(node, 0)
    
    # Iterative
    def find_depth_v2(self, root: TreeNode) -> int:
        stack = []
        # push just like callstack
        if root is not None:
            stack.append((1,root))
        
        depth = 0
        while stack != []:
            current_depth, root = stack.pop()
            if root is not None:
                depth = max(depth, current_depth)
                stack.append((current_depth+1, root.left))
                stack.append((current_depth+1, root.right))
        return depth
    
    
a = TreeNode(0)
a.left = TreeNode(1)
a.right = TreeNode(0)
a.right.left = TreeNode(1)
a.right.right = TreeNode(0)
a.right.left.left = TreeNode(1)
a.right.left.right = TreeNode(1)
a.right.left.right.right = TreeNode(1)
a.right.left.right.right.right = TreeNode(1)
a.right.left.right.right.right.right = TreeNode(1)
a.right.left.right.right.right.right.right = TreeNode(1)

print(Solution().find_depth(a))