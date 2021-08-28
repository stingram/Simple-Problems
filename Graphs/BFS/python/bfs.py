from typing import Dict, Set

class Graph(object):
    def __init__(self):
        self.V: int = 0
        self.G: Dict[int, Set[int]] = {}
    
    def add_node(self, a: int):
        if a not in self.G:
            self.G[a] = set()
    
    def add_edge(self, a: int, b: int):
        self.G[a].add(b)
        

    def __str__(self):
        out = "\n"
        for i in self.G:
            out += f"Node: {i}, Children: {self.G[i]}\n"
        return out
    
    def _bfs_helper(self, visited: Set[int], i: int):
        n_queue: List[int] = []
        visited.add(i)
        n_queue.append(i)
        
        while len(n_queue) > 0:
            s = n_queue.pop()
            print(f"Visited Node: {s}.")
            for n in self.G[s]:
                if n not in visited:
                    n_queue.append(n)
                    visited.add(n) 
    
    def bfs(self):
        visited: Set[int] = set()
        for i in self.G.keys():
            if i not in visited:
                self._bfs_helper(visited, i)
g = Graph()
g.add_node(0)
g.add_node(1)
g.add_node(2)

g.add_edge(0,1)
g.add_edge(0,2)


print(f"Graph: {g}")
g.bfs()