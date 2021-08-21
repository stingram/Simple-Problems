
class Graph(object):
    def __init__(self, v):
        self.graph = [[0 for column in range(v)] for row in range(v)]
        self.V = v
        
    def min_dist(self, visited, dists):
        min_dist = float("inf")
        min_node = None
        for i,n in enumerate(dists):
            if i not in visited:
                if dists[i] < min_dist:
                    min_dist = n
                    min_node = i
                    print("HERE")
        return min_node

    def _print_sol(self, prev, dists):
        print(f"Vertex \tDistance from Source")
        for node in range(self.V):
            print(f"{node}, \t {dists[node]}")

        print(f"Vertex \tParent")
        for node in range(self.V):
            print(f"{node}, \t {prev[node]}") 

    def dijkstra(self, start):
        dists = [float("inf")]* self.V
        prev = {start: None}
        visited = {}
        dists[0] = 0
        while(len(visited)<self.V):
            curr = self.min_dist(visited, dists)
            print(f"Curr: {curr}")
            for n in range(self.V):
                if self.graph[curr][n] != 0 and n not in visited: # 0 means not connected
                    # get new dist
                    dist_n = dists[curr]+self.graph[curr][n]
                    if dist_n < dists[n]:
                        print(f"HERE2: {n}, dist: {dist_n}")
                        dists[n] = dist_n
                        prev[n] = curr
            visited[curr] = True
            
        self._print_sol(prev, dists)











# Driver program
g = Graph(9)
g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
        [4, 0, 8, 0, 0, 0, 0, 11, 0],
        [0, 8, 0, 7, 0, 4, 0, 0, 2],
        [0, 0, 7, 0, 9, 14, 0, 0, 0],
        [0, 0, 0, 9, 0, 10, 0, 0, 0],
        [0, 0, 4, 14, 10, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 1, 6],
        [8, 11, 0, 0, 0, 0, 1, 0, 7],
        [0, 0, 2, 0, 0, 0, 6, 7, 0]
        ];
g.dijkstra(0);