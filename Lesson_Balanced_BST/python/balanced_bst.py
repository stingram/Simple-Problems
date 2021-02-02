class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
        
        
def is_balanced_helper(node: Node, level: int):
    if node.left is None and node.right is None:
        return [True, level]
    
    left_depth = level
    right_depth = level
    
    if node.left:
        l_balance, left_depth = is_balanced_helper(node.left, level+1)
        if not l_balance:
            return [False, left_depth]
    if node.right:
        r_balance, right_depth = is_balanced_helper(node.right, level+1)
        if not r_balance:
            return [False, right_depth]

    diff = abs(left_depth - right_depth) 

    if(diff > 1):   
        return [False, max(left_depth,right_depth)]
    return [True, max(left_depth,right_depth)]

            
def is_balanced(root: Node):
    return is_balanced_helper(root, 1)[0]


def is_balanced_helper_v2(node: Node):
    if node is None:
        return 0
    
    left_depth = is_balanced_helper_v2(node.left)
    right_depth = is_balanced_helper_v2(node.right)

    if left_depth >= 0 and right_depth >= 0 and abs(left_depth-right_depth) <= 1:
        return max(left_depth,right_depth) + 1
    return -1

def is_balanced_v2(node: Node):
    return is_balanced_helper_v2(node) != -1

n4 = Node(4)
n3 = Node(3)
n2 = Node(2, n4)
n1 = Node(1, n2, n3)

#      1
#     / \
#    2   3
#   /
#  4
print(is_balanced(n1))
# True

n4 = Node(4)
n2 = Node(2, n4)
n1 = Node(1, n2, None)

#      1
#     /
#    2
#   /
#  4
print(is_balanced(n1))
# False