
from typing import List
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def __str__(self):
        result = ''
        result += str(self.value)
        if self.left:
            result += str(self.left)
        if self.right:
            result += str(self.right)
        return result

class Solution(object):
    def serialize(self, node: Node) -> str:
        if node is None:
            return "#"
        return str(node.value) + ' ' + self.serialize(node.left) + " " + self.serialize(node.right)
    
    def deserialize(self, ser_string: str) -> Node:
        def helper(values):
            value = next(values)
            
            if value == "#":
                return None
            node = Node(int(value))
            node.left = helper(values)
            node.right = helper(values)
            
            return node
        
        values = iter(ser_string.split())
        return helper(values)

    
root = Node(1)
root.left = Node(3)
root.right = Node(4)
root.left.left = Node(2)
root.left.right = Node(5)
root.right.right = Node(7)

serial = Solution().serialize(root)
print(serial)

res_node = Solution().deserialize(serial)
print(res_node)
