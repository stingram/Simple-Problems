# Given a binary tree, extract all the leaves in repeated succession
# into a list of lists by starting at the bottom and working your way upwards.
from typing import List, Dict

class Node:
    def __init__(self,val: int, left: 'Node' = None, right: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right

def _traverse_graph(node, prev_mapper, children_counter):
    if node is None:
        return
    num_children = 0
    if node.left:
        num_children += 1
        prev_mapper[node.left]=node
    if node.right:
        num_children += 1
        prev_mapper[node.right]=node
    if num_children == 0:
        children_counter[num_children].append(node)
    else:
        children_counter[num_children].add(node)
    
    _traverse_graph(node.left,prev_mapper,children_counter)
    _traverse_graph(node.right,prev_mapper,children_counter)

def extract_leaves(root: Node)-> List[List[int]]:
    res = []
    
    # store prev_mapper and children_counter so we only have to 
    # traverse graph once
    prev_mapper = {}
    children_counter: Dict[int,List[int]] = {0: [], 1: set(), 2: set()}
    
    # traverse graph
    prev_mapper[root] = -1
    _traverse_graph(root, prev_mapper,children_counter)
    
    # use children_counter and prev_mapper to successively
    # append leaves to result in linear time
    i = 0
    parent = 0
    while parent != -1:
        num_leaves = len(children_counter[0][i:])
        curr_leaves = children_counter[0][i:]
        res.append([cl.val for cl in curr_leaves])
        for leaf in curr_leaves:
            parent = prev_mapper[leaf]
            
            # check if this parent has 2 or 1 children in constant time
            # and update children_counter accordingly
            if parent in children_counter[2]:
                children_counter[2].remove(parent)
                children_counter[1].add(parent)
            elif parent in children_counter[1]:
                children_counter[1].remove(parent)
                children_counter[0].append(parent)
        i+= num_leaves

    return res

root = Node(5,Node(9,Node(44),Node(14,Node(6),Node(2))),Node(7))

print(f"{extract_leaves(root)}")