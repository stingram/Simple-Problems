from typing import List
from collections import defaultdict

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]):
        
        # if something that isn't initialized, this makes an empty list
        graph = defaultdict(list)
        
        # build our graph from our list of prerequisites
        for edge in prerequisites:
            graph[edge[0]].append(edge[1])
            
        # at this point, we'll have a map from each vertex to a list of all other connected vertices
        # This is just an adjacency list
        
        # This is our visited list
        visited = set()

        # Returns True if there is a cycle
        def visit(vertex):
            visited.add(vertex)
            for neighbor in graph[vertex]:
                # If we are seeing a node again in this set we have a cycle
                # We also have a cycle if this neighbor has a cycle
                if neighbor in visited or visit(neighbor):
                    return True
            # Need to empty this sequence of visited nodes
            visited.remove(vertex)
            return False
        
        # Call on every node
        for i in range(numCourses):
            if visit(i):
                return False
        return True

        
    def canFinishKahn(self, n: int, prereqs: List[List[int]]):
        
        # create adjaceny list
        G = defaultdict(list)
        for prereq in prereqs:
            G[prereq[0]].append(prereq[1])
            
        # Create in_degree count
        in_degree = [0]*n
        
        # for each node
        for i in range(n):
            # for nodes in adjaceny list add one to in_degree
            for j in G[i]:
                in_degree[j] += 1
                
        # queue
        queue = []
                
        # only add nodes that have zero in_degree
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)
                
        # kahn algo for cycle detection 
        cnt = 0
        while(queue):
            
            # pop node
            node = queue.pop(0)
            
            # check all it's neighbors, subtract 1 from them
            for neighbor in G[node]:
                in_degree[neighbor] -= 1
                
                # check if we need to add it the queue
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            
            # increment cnt
            cnt += 1
            
        if cnt == n:
            return False    # No cycle
        else:
            return True     # Cycle found
            
            
            
        
n = 4
prereqs = []
# prereqs.append([0, 1])
prereqs.append([0, 1])
prereqs.append([1, 2])
prereqs.append([2, 3])
prereqs.append([3, 0])

print(not Solution().canFinishKahn(n, prereqs))