class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def findLargestBSTSubtree(root):
    # Helper function to traverse the tree and return information about each subtree
    def traverse(node):
        # Base case: An empty subtree is a BST of size 0
        if not node:
            return (True, 0, float('inf'), float('-inf'))
        
        # Recursively check the left and right subtrees
        leftIsBST, leftSize, leftMin, leftMax = traverse(node.left)
        rightIsBST, rightSize, rightMin, rightMax = traverse(node.right)
        
        # Check if the current node forms a BST with its left and right subtrees
        if leftIsBST and rightIsBST and leftMax < node.value < rightMin:
            # The current subtree is a BST
            return (True, 1 + leftSize + rightSize, min(node.value, leftMin), max(node.value, rightMax))
        else:
            # The current subtree is not a BST, return the size of the largest BST found so far in the subtrees
            return (False, max(leftSize, rightSize), 0, 0)
    
    # The size of the largest BST is the second element of the tuple returned by traverse(root)
    _, largestBSTSize, _, _ = traverse(root)
    return largestBSTSize

# Example Usage
# root = TreeNode(10, 
#                 TreeNode(5, 
#                          TreeNode(1), 
#                          TreeNode(8)), 
#                 TreeNode(15, 
#                          None, 
#                          TreeNode(7)))
# print(findLargestBSTSubtree(root))

tree = TreeNode(2, TreeNode(1,None,TreeNode(4,TreeNode(3),TreeNode(6, TreeNode(5),TreeNode(5))))) # should be 5
print(f"largest: {findLargestBSTSubtree(tree)}")