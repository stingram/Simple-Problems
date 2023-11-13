# Description: The thief has found himself a new place for his thievery again. 
# There is only one entrance to this area, called the "root." Besides the root,
# each house has one and only one parent house. After a tour, the smart thief
# realized that "all houses in this place forms a binary tree". It will
# automatically contact the police if two directly linked houses were broken
# into on the same night.

from typing import Tuple

class TreeNode:
    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val: int = val
        self.left: TreeNode = left
        self.right: TreeNode = right
        
def robHelper(node: TreeNode, odd_sum: int, even_sum: int, level: int) -> Tuple[int,int]:
    if node is None:
        return (odd_sum,even_sum)
    # update current level
    if level % 2 == 0:
        even_sum += node.val
    else:
        odd_sum += node.val
        
    # update values from left
    odd_sum, even_sum = robHelper(node.left,odd_sum,even_sum,level+1)
    
    # update values from right
    odd_sum, even_sum = robHelper(node.right,odd_sum,even_sum,level+1)
    
    # return updated values
    return (odd_sum, even_sum)
            
def rob(root: TreeNode) -> int:
    # determine sum of all levels
    odd_sum, even_sum = robHelper(root,0,0,0)
    
    # return max
    return max(odd_sum,even_sum)


#      3
#     / \
#    2   3
#     \   \ 
#      3   1

# Ans: 7 

root = TreeNode(3)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.right = TreeNode(3)
root.right.right = TreeNode(1)

print(f"Max sum that can be robbed is: {rob(root)}.")