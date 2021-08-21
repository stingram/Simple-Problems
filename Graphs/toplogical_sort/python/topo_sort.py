# THIS IS IF THE NODE ID's ARE STRINGS

# TIME IS O(V+E)
# SPACE IS O(V)

from collections import defaultdict

class Graph(object):
	def __init__(self, V):
		self.graph = defaultdict(list)
		self.V = V
		
	def add_edge(self, u: str, v: str):
		self.graph[u].append(v)
		
		
# topological sort
def top_sort(g: Graph):
    # THIS IS IF THE NODE ID's ARE STRINGS
	in_degree = defaultdict(int)
 
    # THIS IS IF THE NODE ID'S ARE INTEGERS
    # in_degree = [0]*(g.V)
  
	
	for i in g.graph:
		for j in g.graph[i]:
			in_degree[j] += 1
	
	queue = []
	for i in g.graph:
		if in_degree[i] == 0:
			queue.append(i)
			
	cnt = 0
	schedule = []
	while(queue):
		u = queue.pop(0)
		schedule.append(u)
		
		for n in g.graph[u]:
			in_degree[n] -= 1
			if in_degree[n] == 0:
				queue.append(n)
				
		cnt += 1
		
	if cnt != g.V:
		# CYCLE ! :(
		return []
		
	return schedule
		
  
g = Graph(3)
g.add_edge("A", "B")
g.add_edge("B", "C")
g.add_edge("A", "C")

print(top_sort(g))


g = Graph(5)
g.add_edge("A", "C")
g.add_edge("A", "B")
g.add_edge("C", "B")
g.add_edge("B", "D")
g.add_edge("C", "D")
g.add_edge("C", "E")

print(top_sort(g))