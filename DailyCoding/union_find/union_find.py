class UnionFind:
    def __init__(self):
        self.id = {}
    def find(self, x):
        y =self.id.get(x,x)
        if y != x:
            self.id[x] = y = self.find(y)
        return y
        
    def union(self,x,y):
        self.id[self.find(x)] = self.find(y)
        
# Create an instance of UnionFind
uf = UnionFind()

# Perform union operations
uf.union(1, 2)
uf.union(3, 4)

# Find representatives
print(uf.find(1))  # Output: Representative of the set {1, 2, 3, 4}
print(uf.find(2))  # Output: Representative of the set {1, 2, 3, 4}
print(uf.find(3))  # Output: Representative of the set {1, 2, 3, 4}
print(uf.find(4))  # Output: Representative of the set {1, 2, 3, 4}

uf.union(1, 3)

# Find representatives
print(uf.find(1))  # Output: Representative of the set {1, 2, 3, 4}
print(uf.find(2))  # Output: Representative of the set {1, 2, 3, 4}
print(uf.find(3))  # Output: Representative of the set {1, 2, 3, 4}
print(uf.find(4))  # Output: Representative of the set {1, 2, 3, 4}