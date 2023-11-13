# Description: Given two binary trees s and t, check if tree t
# is a subtree of tree s. A subtree of s is a tree consisting of
# a node in s and all of its descendants.

#      3
#     / \
#    4   5
#   / \
#  1   2

#    4
#   / \
#  1   2

class TreeNode:
    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right


# find root of t in s
def find_root(s: TreeNode, t: TreeNode) -> TreeNode:
    if s is None or t is None:
        return None
    q = [s]
    while q:
        curr_code = q.pop(0)
        if curr_code is None:
            continue
        if curr_code.val == t.val:
            return curr_code
        q.append(curr_code.left)
        q.append(curr_code.right)
    return None


# traverse both trees
def check_subtree(s: TreeNode, t: TreeNode) -> bool:
    # base case
    if t is None and s is None:
        return True
    
    if t is None and s is not None:
        return False
    
    if t is not None and s is None:
        return False
    
    # if they match
    if t.val == s.val:
        # check left
        check_subtree_left = check_subtree(t.left, s.left)
        # check right
        check_subtree_right = check_subtree(t.right, s.right)
        
        if check_subtree_left and check_subtree_right:
            return True
        
    return False
    


def isSubtree(s: TreeNode, t: TreeNode) -> bool:
    sub_root = find_root(s,t)
    if sub_root is None:
        return False
    return check_subtree(sub_root, t)

s = TreeNode(3)
s.left = TreeNode(4)
s.right = TreeNode(5)
s.left.left = TreeNode(1)
s.left.right = TreeNode(2)

t = TreeNode(4)
t.left = TreeNode(1)
t.right = TreeNode(2)

print(f"t is a subtree of s? {isSubtree(s,t)}.")