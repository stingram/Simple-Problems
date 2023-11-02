# Given the root of a binary tree, invert the tree, 
# swapping every left and right child of nodes.

#      4
#    /   \
#   2     7
#  / \   / \
# 1   3 6   9

class TreeNode:
    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right
        
        
def invert_tree(node: TreeNode) -> TreeNode:
    # base case
    if node is None:
        return None
    
    # invert left and right sides
    inverted_left = invert_tree(node.left)
    inverted_right = invert_tree(node.right)
    
    #swap nodes
    node.right = inverted_left
    node.left = inverted_right
    
    # return node
    return node


def print_tree(node: TreeNode):
    if not node:
        return
    
    print_tree(node.left)
    print_tree(node.right)
    print(f"val: {node.val}")

#      4
#    /   \
#   2     7
#  / \   / \
# 1   3 6   9
root = TreeNode(4)
root.left = TreeNode(2)
root.left.left = TreeNode(1)
root.left.right = TreeNode(3)

root.right = TreeNode(7)
root.right.left = TreeNode(6)
root.right.right = TreeNode(9)

print_tree(root)
print_tree(invert_tree(root))

