import numpy as np

def knn(query, candidates, k):
    diff = candidates - query
    print(f"{diff}")
    distance = np.sum(diff**2,axis=-1)
    print(f"{distance}")
    res = candidates[np.argsort(distance)[:k]]
    print(f"{res}")
    
    

def knnv2(query, candidates, k):
    distances = np.linalg.norm(candidates-query, axis=1)
    return candidates[np.argsort(distances)[:k]]
    
candidates = np.arange(10,dtype=np.float32).reshape((5,2))
query = np.array([[4.5,1.22]])
k = 2

knn(query, candidates, k)

print(f"{knnv2(query, candidates, k)}")