from typing import List

class Node:
    def __init__(self,val,left=None,right=None) -> None:
        self.val = val
        self.left = left
        self.right = right

def compute_boundary(root: Node) -> List[int]:
    if not root:
        return [-1]
    
    res = [root.val]
    
    # left boundary
    curr = root
    while curr:
        if curr != root and (curr.left is not None or curr.right is not None):
            res.append(curr.val)
        if curr.left:
            curr = curr.left
        else:
            curr=curr.right

    # right boundary
    curr = root
    right_stack = []
    while curr:
        if curr != root and (curr.left is not None or curr.right is not None):
            right_stack.append(curr.val)
        if curr.right:
            curr = curr.right
        else:
            curr = curr.left

    stack = [root]
    leaves = []
    while stack:
        curr = stack.pop()
        if curr != root and curr.left is None and curr.right is None:
            leaves.append(curr.val)
        if curr.right:
            stack.append(curr.right)
        if curr.left:
            stack.append(curr.left)
            
    return res+leaves+right_stack[::-1]


# [1,2,4,5,6,8,7,3]
root = Node(1,Node(2,Node(4,None,Node(5))),Node(3,Node(6),Node(7,None,Node(8))))
print(f"res:{compute_boundary(root)}")

# [1,2,4,5,3]
root = Node(1,Node(2,Node(4)),Node(3,Node(5)))
print(f"res:{compute_boundary(root)}")

# [1,2,4,6,7,3]
root = Node(1,Node(2,Node(4),Node(5,Node(6),Node(7))),Node(3))
print(f"res:{compute_boundary(root)}")

# [1,2,4,5,3]
root = Node(1,Node(2,Node(4)),Node(3,None,Node(5)))
print(f"res:{compute_boundary(root)}")

# [1,3,2]
root = Node(1,None,Node(2,None,Node(3)))
print(f"res:{compute_boundary(root)}")

# [1,2,3,5,4]
root = Node(1,Node(2,Node(3),Node(6)),Node(4,None,Node(5)))
print(f"res:{compute_boundary(root)}")