class Node(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return self.val
    
    
class Solution(object):
    def get_depth(self, root: Node, count: int) -> int:
        # base case
        if not root:
            return count
        
        # Got to this node so increment our count
        count += 1
        
        # in order traversal
        count_left = self.get_depth(root.left, count)
        
        # node
        count_right = self.get_depth(root.right, count)

        return max([count, count_left, count_right])

root = Node('a')
root.left = Node('b')
root.left.left = Node('d')
root.left.left.right = Node('e')
root.left.left.right.left = Node('f')
root.left.left.right.left.right = Node('g')
root.right = Node('c')
print(Solution().get_depth(root, 0))