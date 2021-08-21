class Node(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class Solution(object):
    def unival_subtree(self, node) -> int:
        def unival_helper(node, subtrees):
            
            # No Node
            if node is None:
                return (True, subtrees)
            
            # leaf Node
            if node.left is None and node.right is None:
                return (True, subtrees + 1)
            
            rl = unival_helper(node.right, subtrees)
            subtrees = rl[1]
            rr = unival_helper(node.left, subtrees)
            subtrees = rr[1]
            
            # left child only
            if node.left and not node.right:
                if rl[0] and node.left.val == node.val:
                    return (True, subtrees + 1)
            
            # right child only
            if node.right and not node.left:
                if rr[0] and node.right.val == node.val:
                    return (True, subtrees + 1)
            
            # both
            if rr[0] and  rl[0] \
                and node.right.val == node.val and node.left.val == node.val:
                    return (True, subtrees + 1)
            return (False, subtrees)
        
        return unival_helper(node, 0)[1]
    
    
    # CODERPRO VERSION
    def count_unival_subtrees(self, node):
        count, is_unival = self.count_unival_helper(node)
        return count
    
    def count_unival_helper(self, node):
        if not node:
            return 0, True
        left_count, is_left_unival = self.count_unival_subtrees(node.left)
        right_count, is_right_unival = self.count_unival_subtrees(node.right)
        
        if is_left_unival and is_right_unival and (not node.left or node.val == node.left.val) and (not node.right or node.val == node.right.val):
            return left_count + right_count + 1, True
        return left_count + right_count, False
    
    
    
#    0
#   / \
#  1   0
#     / \
#    1   0
#   / \
#  1   1
a = Node(0)
a.left = Node(1)
a.right = Node(0)
a.right.left = Node(1)
a.right.right = Node(0)
a.right.left.left = Node(1)
a.right.left.right = Node(1)

print(Solution().unival_subtree(a))
# 5