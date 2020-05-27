class Node():
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class Solution(object):
    def get_values_at_height(self, root, height):
        vals = []
        def traverse(node, height, current_height, vals):
            if current_height < height:
                if node.left is not None:
                    traverse(node.left, height, current_height+1, vals)
                if node.right is not None:
                    traverse(node.right, height, current_height+1, vals)
            elif current_height == height:
                vals.append(node.value)        
        traverse(node, height, 1, vals)
        return vals
    
    
    def get_values_v2(self, node, depth):
        if not node:
            return []
        
        if depth == 1:
            return [node.value]
        
        return self.get_values_v2(node.left, depth -1) + self.get_values_v2(node.right, depth -1)
    
#    1
#   / \
#  2   3
# / \   \
# 4   5   7
node = Node(1)
node.left = Node(2)
node.right = Node(3)
node.right.right = Node(7)
node.left.left = Node(4)
node.left.right = Node(5)
height = 3
print(Solution().get_values_at_height(node, height))