from typing import List, Dict, Optional

class Node:
    def __init__(self, val: int, neighbors: Optional[List['Node']] = None):
        self.val = val
        self.neighbors: List['Node'] = neighbors if neighbors is not None else []
    # def __repr__(self):
    #     return f"Node: {self.val}. Neighbors: {self.neighbors}"
        
        
def _clone_graph(node: Node, visited: Dict[Node,Node]) -> Node:
    # base case, empty node
    if not node:
        return None
    
    # node is already visited
    if node in visited:
        return visited[node]
    
    # create new node
    new_node = Node(node.val)
    
    # add this node to visited
    visited[node] = new_node
    
    # add neighbors
    new_node.neighbors =  [_clone_graph(neighbor,visited) for neighbor in node.neighbors]
        
    # return new_node
    return new_node

    

def clone_graph(node: Node) -> Node:
    visited: Dict[Node,Node] = {}
    
    return _clone_graph(node,visited)

def printGraph(node: 'Node', visited: Dict['Node', 'Node']):
    if not node or node in visited:
        return

    print(f"Node {node.val} has neighbors: {[neighbor.val for neighbor in node.neighbors]}")

    visited[node] = True

    for neighbor in node.neighbors:
        printGraph(neighbor, visited)


g = Node(0, [Node(1),Node(2, [Node(3),Node(4)])])

new_g = clone_graph(g)

# Print the original and cloned graphs
print("Original Graph:")
printGraph(g, {})
print("Cloned Graph:")
printGraph(new_g, {})


node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node1.neighbors = [node2, node3]
node2.neighbors = [node1]
node3.neighbors = [node1]

new_g = clone_graph(node1)

print("Original Graph:")
printGraph(node1, {})
print("Cloned Graph:")
printGraph(new_g, {})