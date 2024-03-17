# two nodes in BST have been swapped, recover the tree!

class Node:
    def __init__(self,val,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.val}"

def _find_first_wrong_node(node):
    if not node:
        return None
    
    # has two children
    if node.left and node.right:
        if node.left.val < node.val and node.right.val > node.val:
            # search children
            bad_node = _find_first_wrong_node(node.left)
            if bad_node:
                return bad_node
            return _find_first_wrong_node(node.right)
        else:
            # found the bad node
            return node

    # has left only
    elif node.left and node.right is None:
        if node.left.val < node.val:
            # bad node must be on left side
            return _find_first_wrong_node(node.left)
        else:
            # found the bad node
            return node
    
    # has right only
    elif node.left is None and node.right:
        if node.right.val > node.val:
            # bad node must be on right side
            return _find_first_wrong_node(node.right)
        else:
            # found bad node
            return node    
    # has no children - just return None 
    else:
        return None

def _valid_node(node):
    if node is None:
        return True
    if node.left and node.right and node.left.val < node.val and node.right.val > node.val:
        return True
    if node.left and node.right is None and node.left.val < node.val:
        return True
    if node.right and node.left is None and node.right.val > node.val:
        return True
    if node.left is None and node.right is None:
        return True
    return False
    
def _check_valid_swap(node1, node2):
    if node1 is None or node2 is None:
        return False
    
    if node1.left == node2:
        new_node1 = Node(node2.val,node1,node1.right)
    elif node1.right == node2:
        new_node1 = Node(node2.val,node1.left,node1)
    else:    
        new_node1 = Node(node2.val,node1.left,node1.right)
    new_node2 = Node(node1.val,node2.left,node2.right)
    
    return _valid_node(new_node1) and _valid_node(new_node2)

def _helper(first,curr):
    if curr is None:
        return None
        
    # check if swapping would be valid
    if _check_valid_swap(first, curr):
        
        # it is valid to swap them
        temp = curr.val
        curr.val = first.val
        first.val = temp
        return first
    else: # need to swap some other node
        swapped = _helper(first,curr.left)
        if not swapped:
            swapped = _helper(first,curr.right)
        return swapped

def recover_tree(root):
    # get first wrong node
    first = _find_first_wrong_node(root)
       
    # traverse descendants until found and swap nodes
    if first is not None:
        swapped = _helper(first,first.left)
        if not swapped:
            swapped = _helper(first,first.right)
    # return
    


tree = Node(1,Node(3,None,Node(2)))

recover_tree(tree)
print(f"{tree},{tree.left}, {tree.right}, {tree.left.right}")