# TODO - FINISH

# Description: You are given a binary tree in which each node contains
# an integer value. Find the number of paths that sum to a given value.
# The path does not need to start or end at the root or a leaf but must
# go downwards (traveling only from parent nodes to child nodes).

# The binary tree has a unique value in each node, and
# an additional targetSum parameter is provided.

from typing import Dict, Tuple, List

class TreeNode:
    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val: int = val
        self.left: TreeNode = left
        self.right: TreeNode = right

def calcualte_paths_sums(node: TreeNode, current_sum: int, target: int, counts: Dict[int,int]) -> List[List[int]]:
    
    # base case
    if node == None:
        return 0
    
    # update running sum
    current_sum += node.val

    # Check how many paths end at the current node and sum to the target
    complement = current_sum - target
    total_paths = counts.get(complement, 0)
    
    # update the dictionary with the current running sum
    counts[current_sum] = counts.get(current_sum,0) + 1
    
    # recursively explore left and right subtrees
    total_paths += calcualte_paths_sums(node.left, current_sum, target, counts)
    total_paths += calcualte_paths_sums(node.right, current_sum, target, counts)
    
    # Decrement the frequency of the curent running sum
    counts[current_sum] -= 1
    
    return total_paths

#      10
#     /  \
#    5   -3
#   / \    \
#  3   2   11
# / \   \
# 20  -2   1

# The number of paths that sum to 8 is 3, which corresponds to the paths: 
# 5 -> 3, 5 -> 2 -> 1, and -3 -> 11.

root = TreeNode(10)
root.left = TreeNode(5)
root.left.left = TreeNode(3)
root.left.left.left = TreeNode(20)
root.left.left.right = TreeNode(-2)

root.left.right = TreeNode(2)
root.left.right = TreeNode(1)

root.right = TreeNode(-3)
root.right.right = TreeNode(11)

target = 8
print(f"Num paths that sum to {target} is {calcualte_paths_sums(root,target,0)[1]}")