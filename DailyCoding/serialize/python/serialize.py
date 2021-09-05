from __future__ import annotations

class Node(object):
    def __init__(self, val: int, left: Node = None, right: Node = None):
        self.val = val
        self.left = left
        self.right = right
    
    # For convenience
    def __str__(self):
        res = ""
        res += str(self.val)
        if self.left is not None:
            res += str(self.left)
        if self.right is not None:
            res += str(self.right)
        return res
        
class Solution(object):
    def serialize(self, node: Node) -> str:
        # Need marker for None Node type
        if node is None:
            return "#"
        
        # We will serialize with pre-order traversal, which is root, left, right
        # We want space between node values so that we can deserialize
        return str(node.val) + " " + self.serialize(node.left) + " " + self.serialize(node.right)
    
    def _helper(self, values):
        val = next(values)
        
        if val == '#':
            return None

        node = Node(int(val))
        node.left = self._helper(values)
        node.right = self._helper(values)
        return node
    
    def deserialize(self, ser_string: str) -> Node:
        # convert string to list
        ser_split = ser_string.split()
        
        # create iterator object
        values = iter(ser_split)
        
        # use helper to deserialize
        return self._helper(values)
    
    
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)

print(root)
serial = Solution().serialize(root)
print(serial)
print(Solution().deserialize(serial))