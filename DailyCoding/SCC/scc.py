from typing import List, Tuple, Set, Dict

class SCC:
    def __init__(self,graph) -> None:
        self.graph = graph
        self.n_nodes = len(self.graph.keys())
        self.UNVISITED = -1
        self.ids=[self.UNVISITED]*self.n_nodes
        self.low=[0]*self.n_nodes
        self.on_stack = [False]*self.n_nodes
        self.stack = []
        self.id_val = 0
        self.num_sccs = 0

    def _dfs(self,node):
        print(f"node: {node}. ids: {self.ids}")
        self.ids[node]=self.id_val
        self.low[node]=self.id_val
        self.id_val += 1
        self.stack.append(node)
        self.on_stack[node] = True

        for neighbor in self.graph[node]:
            # visit neighbors
            if self.ids[neighbor] == self.UNVISITED:
                self._dfs(neighbor)
            # set low
            if self.on_stack[neighbor]:
                self.low[node] = min(self.low[node],self.low[neighbor])

        # once here, node has the lowest id possible
        # now we set whatever is on top of it on the
        # stack to have the same low value

        # check if we're done with this SCC
        if self.ids[node] == self.low[node]:
            # pop nodes on the stack until we get to this node
            while True:
                n = self.stack.pop()
                self.on_stack[n] = False
                self.low[n]=self.ids[node]
                if n == node:
                    break
            self.num_sccs += 1

    def find_SCCs(self):
        for node in self.graph.keys():
            print(f"NODE: {node}")
            if self.ids[node] == self.UNVISITED:
                self._dfs(node)

    def create_scc_map(self):
        scc_map = {}
        for i,val in enumerate(self.low):
            if val not in scc_map:
                scc_map[val] = []
            scc_map[val].append(i)
        return scc_map

def create_scc_graph(graph, scc_map):
    scc_graph = {}
    return scc_graph

# should create 3 SCCs
graph = {0:[1],
         1:[6,2,4],
         2:[3],
         3:[2,4,5],
         4:[5],
         5:[4],
         6:[0,2]}
scc = SCC(graph)
scc.find_SCCs()
print(f"{scc.low}.")
print(f"num SCCs: {scc.num_sccs}.")
print(f"map: {scc.create_scc_map()}")