from typing import List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
# Create the binary tree
#        3
#       / \
#      5   1
#     / \ / \
#    6  2 0  8
#      / \
#     7   4

# LCA of nodes 5 and 1 is TreeNode(3)
# LCA of nodes 5 and 4 is TreeNode(5)
# LCA of nodes 2 and 0 is TreeNode(2)
def LCA(root:TreeNode, p:TreeNode, q:TreeNode) -> TreeNode:
        if root == None or root == p or root == q:
            return root
        
        # search sub trees
        left = LCA(root.left,p,q)
        right = LCA(root.right,p,q)
        
        # both are found then return this node
        if left and right:
            return root
        
        # if only left found
        if left:
            return left
        
        #if only right found
        if right:
            return right


t = TreeNode(3)
t.left = TreeNode(5)
t.right = TreeNode(1)
t.left.left = TreeNode(6)
t.left.right = TreeNode(2)

t.right.left = TreeNode(0)
t.right.right = TreeNode(8)

t.left.right.left = TreeNode(7)
t.left.right.right = TreeNode(4)

print(f"LCA of nodes 4 and 5 is: {LCA(t,t.left,t.left.right.right).val}.")