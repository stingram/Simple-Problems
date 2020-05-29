class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        
class Solution:
    def invert(self, node: Node) -> Node:
        if node is None:
            return
        node.left = self.invert(node.left)
        node.right = self.invert(node.right)
        
        temp = node.left
        node.left = node.right
        node.right = temp
        return node
    
    
    
root = Node(1)

root.left = Node(2)
root.right = Node(3)

root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)

inverted = Solution().invert(root)

print(root.val)

print(root.left.val) # 2
print(root.right.val)

print(root.left.left.val)
print(root.left.right.val)
print(root.right.left.val)
print(root.right.right.val)

