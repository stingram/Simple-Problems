from typing import List, Dict, Set, Optional

class Node:
    def __init__(self, val: int, neighbors: Optional[List['Node']] = None):
        self.val = val
        self.neighbors = neighbors if neighbors else []


def print_graph(node: Node):
    if not node:
        print(f"Empty graph!")
        return

    visited: Set[Node] = set()

    queue: List[Node] = [node]
    while queue:
        curr_node = queue.pop(0)
        print(f"Node: {curr_node.val}. Neighbors: {[neighbor.val for neighbor in curr_node.neighbors]}")
        visited.add(curr_node)
        for neighbor in curr_node.neighbors:
            if neighbor not in visited:
                queue.append(neighbor)        


def _clone_graph(node: Node, visited: Dict[Node,Node]) -> Node:
    
    # check if we are at a left
    if not node:
        return None
    
    # have we cloned this node already?
    if node in visited:
        return visited[node]
    
    # we need to clone this node
    new_node = Node(node.val)
    
    # add to visited
    visited[node] = new_node
    
    # add neighbors 
    new_node.neighbors = [_clone_graph(neighbor, visited) for neighbor in node.neighbors]

    # return new node
    return visited[node]

def clone_graph(node: Node) -> Node:
    visited: Dict[Node,Node] = {} 
    return _clone_graph(node, visited)

g = Node(0, [Node(1),Node(2, [Node(3),Node(4)])])

new_g = clone_graph(g)

# Print the original and cloned graphs
print("Original Graph:")
print_graph(g)
print("Cloned Graph:")
print_graph(new_g)


node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node1.neighbors = [node2, node3]
node2.neighbors = [node1]
node3.neighbors = [node1]

new_g = clone_graph(node1)

print("Original Graph:")
print_graph(node1)
print("Cloned Graph:")
print_graph(new_g)
