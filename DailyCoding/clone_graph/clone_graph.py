from typing import List, Tuple, Dict


class Node:
    def __init__(self, val: int = 0, neighbors: List['Node'] = None):
        self.val: int = val
        self.neighbors: List['Node'] = neighbors if neighbors is not None else []
    def __repr__(self):
        return f"Node: {self.val}, {self.neighbors}"

def cloneGraph(node: 'Node') -> 'Node':
    # base
    if not node:
        return None
    
    # keep track of visited nodes
    v: Dict[Node,Node] = {}
    
    # create root node
    n = Node(node.val)
    
    v[node] = n
    q: List['Node'] = [node]
    
    while q:
        orig_node = q.pop(0)
        
        # Process node - add neighbors
        for neighbor in orig_node.neighbors:
            print(f"HERE")
            if neighbor not in v:
                q.append(neighbor)
                new_neigbor_node = Node(neighbor.val)
                v[neighbor]=new_neigbor_node
                
            # either add new created node or reference to one that already exists
            v[orig_node].neighbors.append(v[neighbor])     
    
    return n


g = Node(0, [Node(1),Node(2, [Node(3),Node(4)])])

new_g = cloneGraph(g)

print(f"Original: {g}.")
print(f"New: {new_g}.\n\n")


node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node1.neighbors = [node2, node3]
node2.neighbors = [node1]
node3.neighbors = [node1]

new_g = cloneGraph(node1)

print(f"Original: {node1}.")
print(f"New: {new_g}.")


# bfs through the original graph making copies of it's node
# in bfs, you pop from the queue and process
# then you process neighbors