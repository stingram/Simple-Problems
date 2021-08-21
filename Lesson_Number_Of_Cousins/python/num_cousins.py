from typing import Dict

class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def __init__(self):
        pass
    
    def _return_vals(self, C: Dict[int,int], n: Node):
        C.pop(n, None)
        return [k.val for k,_ in C.items()]
        
    def list_cousins(self, root: Node, node: Node):
        # we will travel graph until we get to level where node is
        
        # keep track of level, like zigzag, we will process node and cousins in the same iteration
        
        level = 1
        C={}
        N={}
        C[root]=level
        while C:
            while C:
                # process
                
                # we get to the node, so all other nodes in C are the cousins
                if node in C:
                    return self._return_vals(C,node)
                else:
                    # put all children in nodes in C into N dict
                    level += 1
                    for c_node, level in C.items():
                        if c_node.left:
                            N[c_node.left] = level
                        if c_node.right:
                            N[c_node.right] = level
                    C={}
                
                
            C=N
            N={}
        return []
                
    
class Solution_v2(object):
    
    def _nodes_at_height(self, node, height, exclude):
        if node == None or node == exclude:
            return []
        if height == 0:
            return [node.val]
        return (self._nodes_at_height(node.left, height-1, exclude) +
                self._nodes_at_height(node.right, height-1, exclude))
    
    
    def _find_node(self,node, target, parent, height=0):
        if not node:
            return False
        if node == target:
            return (height, parent)
        return (self._find_node(node.left,target, node, height+1) or 
                self._find_node(node.right,target, node, height+1))
    
    def list_cousins(self, node, target):
        height, parent = self._find_node(node, target, None)
        return self._nodes_at_height(node, height, parent)
    
#     1
#    / \
#   2   3
#  / \    \
# 4   6    5
root = Node(1)
root.left = Node(2)
root.left.left = Node(4)
root.left.right = Node(6)
root.right = Node(3)
root.right.right = Node(5)
print(Solution().list_cousins(root, root.right.right))
# [4, 6]

print(Solution_v2().list_cousins(root, root.right.right))
# [4, 6]