# Given the root of a complete binary tree,
# return the number of nodes in the tree.

# For this problem a binary tree is considered “complete”
# if every level besides the last level is completely filled in.

# So if it's complete, return number of nodes, else return -1


# def count_number_of_nodes_v1(node: Node) -> int:
#     if node is None:
#         return 0
#     return count_number_of_nodes_v1(node.left)+count_number_of_nodes_v1(node.right)+1

class Node:
    def __init__(self, val:int, left=None,right=None):
        self.val = val
        self.left = left
        self.right = right


def calculate_height(node: Node) -> int:
    if not node:
        return -1
    height = 0
    while node.left:
        node = node.left
        height += 1
    return height

def count_number_of_nodes(node: Node) -> int:
    # count
    count = 0
    # get hieght of tree
    height = calculate_height(node)
    
    # now check left and right
    while node:
        if calculate_height(node.right) == height - 1:
            # count left this node + left side
            count += 2**height
            node = node.right
        else: # we go left, and height of right is 2 less than current height
            # we have height-1 so we can count the current node
            count += 2**(height-1)
            node = node.left
        # update height
        height -= 1
    
    return count

# Should be 2
root = Node(1,Node(2))
print(f"{count_number_of_nodes(root)}")

# Should be 6
root = Node(1,Node(2,Node(4),Node(5)),Node(3,Node(6)))
print(f"{count_number_of_nodes(root)}")