from math import inf

class Node:
    def __init__(self,val,left = None,right = None):
        self.val = val
        self.left = left
        self.right = right

# return  bounds and size for every subtree
def _helper(node,lower,upper,biggest):
    if node is None:
        return lower, upper, 0
	# get info from from each child
    
    left_lower, left_upper, size_left = _helper(node.left,lower,node.val-1,biggest)
    # print(f"node: {node.val}")
    right_lower, right_upper, size_right = _helper(node.right,node.val+1, upper,biggest)

	# cases for valid bst
	# this node and children are all BSTs
    print(f"node: {node.val}")
    if node.right and node.left and node.val > left_upper and node.val < right_lower:
        
        curr_size = 1 + size_left + size_right 
        if curr_size > biggest[0]:
            biggest[0] = curr_size
        return left_lower, right_upper, curr_size
	
    # this node and only left side
    elif node.left and node.val> left_upper and (node.val > right_lower or not node.right):
        curr_size = 1 + size_left
        if curr_size > biggest[0]:
            biggest[0] = curr_size
        return left_lower, node.val, curr_size

    # this node and only right side
    elif node.right and node.val<right_lower and (node.val < left_upper or not node.left):
        curr_size = 1 + size_right
        if curr_size > biggest[0]:
            biggest[0] = curr_size
        return node.val, right_upper, curr_size

    #  this node only
    else:
        curr_size = 1
        if curr_size > biggest[0]:
            biggest[0] = curr_size
        return node.val, node.val, curr_size

def find_largest_bst(root):
    biggest = [0]
    _helper(root,-inf,inf, biggest)
    return biggest[0]


# tree = Node(2,Node(1), Node(3))
# print(f"largest: {find_largest_bst(tree)}")

# tree = Node(2)
# print(f"largest: {find_largest_bst(tree)}")

# tree = Node(2, Node(3), Node(3))
# print(f"largest: {find_largest_bst(tree)}")


# tree = Node(4,Node(3),Node(5)) # should be 3
# print(f"largest: {find_largest_bst(tree)}")

# tree = Node(2, Node(1,None,Node(4,Node(3),Node(5)))) # should be 3
# print(f"largest: {find_largest_bst(tree)}")


# tree = Node(2, Node(1,None,Node(4,Node(3),Node(6, Node(7),Node(5))))) # should be 3
# print(f"largest: {find_largest_bst(tree)}")


tree = Node(2, Node(1,None,Node(4,Node(3),Node(6, Node(5),Node(5))))) # should be 4
print(f"largest: {find_largest_bst(tree)}")