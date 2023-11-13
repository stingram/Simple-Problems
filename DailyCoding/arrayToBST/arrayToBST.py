# Description: Given an integer array nums sorted in non-decreasing order,
# convert it to a height-balanced binary search tree (BST).

# A height-balanced BST is defined as a binary tree in which the 
# depth of the two subtrees of every node never differs by more than one.

from typing import List

class TreeNode:
    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right

def sortedArrayToBST(nums: List[int]) -> TreeNode:
    if len(nums) == 0:
        return None
    
    mid_point = len(nums)//2
    
    # create node
    node = TreeNode(nums[mid_point])
    
    # build left child
    node.left = sortedArrayToBST(nums[0:mid_point])
    
    # build right child
    node.right = sortedArrayToBST(nums[mid_point+1:])
    
    # return node
    return node

arr = [1,2,3,4,5]

node = sortedArrayToBST(arr)
print(f"Root val: {node.val}")
print(f"Root.left val: {node.left.val}")
print(f"Root.left.left val: {node.left.left.val}")

print(f"Root.right val: {node.right.val}")
print(f"Root.right.left val: {node.right.left.val}")

