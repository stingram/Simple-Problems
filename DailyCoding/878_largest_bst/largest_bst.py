from math import inf

class Node:
	def __init__(self,val,left = None,right = None):
		self.val = val
		self.left = left
		self.right = right

def _helper(node,lower,upper,biggest):
	if node == None:
		return True, lower, upper, 0
	left_bst, left_lower, left_upper, left_size = _helper(node.left,lower,node.val-1,biggest)
	right_bst, right_lower, right_upper, right_size = _helper(node.right,node.val+1,upper,biggest) 

	if not node.left:
		left_lower = node.val
	if not node.right:
		right_upper = node.val
	if node.val > lower and node.val < upper and left_bst and right_bst and ((node.left and node.val>left_upper) or not node.left) and ((node.right and node.val < right_lower) or not node.right):
		curr_size = 1 + left_size + right_size
		if curr_size > biggest[0]:
			biggest[0] = curr_size
		return True, left_lower, right_upper, curr_size
	elif node.val> left_upper and left_bst and node.right is None:
		# only this and left is BST
		curr_size = 1 + left_size
		if curr_size > biggest[0]:
			biggest[0] = curr_size
		return True, left_lower, node.val, curr_size
	elif node.val< right_lower and right_bst and node.left is None: # only this and right is BST
		curr_size = 1 + right_size
		if curr_size > biggest[0]:
			biggest[0] = curr_size
		return True, node.val, right_upper, curr_size
	else: # only this node is BST
		return False, node.val, node.val, 1

def find_largest_bst(root):
	biggest = [0]
	_helper(root,-inf,inf, biggest)
	return biggest[0]


tree = Node(2,Node(1), Node(3)) # should be 3
print(f"largest: {find_largest_bst(tree)}")

tree = Node(2) # should be 1
print(f"largest: {find_largest_bst(tree)}")

tree = Node(2, Node(3), Node(3)) # should be 1
print(f"largest: {find_largest_bst(tree)}")

tree = Node(4,Node(3),Node(5)) # should be 3
print(f"largest: {find_largest_bst(tree)}")

# TODO - Add bigger one
# tree = Node(2, Node(1,None,Node(4,Node(3),Node(5)))) # should be 3
# print(f"largest: {find_largest_bst(tree)}")

# TODO - Add bigger one
# tree = Node(2, Node(1,None,Node(4,Node(3),Node(6, Node(7),Node(5))))) # should be 3
# print(f"largest: {find_largest_bst(tree)}")

# TODO - Add bigger one
# tree = Node(2, Node(1,None,Node(4,Node(3),Node(6, Node(5),Node(5))))) # should be 4
# print(f"largest: {find_largest_bst(tree)}")