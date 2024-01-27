import numpy as np
np.random.seed(42)
def kmeans(input, k, max_iters=100):
    # intialize centroids
    centroids = input[np.random.choice(len(input),size=k,replace=False)]
    print(f"centroids:\n{centroids}")

    for _ in range(max_iters):
        # assign each data point to the nearest centroid
        print(f"input mod:\n{input[:,np.newaxis]}. Shape: {input[:,np.newaxis].shape}")
        distances = np.linalg.norm(input[:,np.newaxis]-centroids, axis=2)
        print(f"distances:\n{distances}")
        
        # make labels
        labels = np.argmin(distances,axis=1)
        print(f"labels: {labels}")

        # update centroids
        new_centroids = np.array([input[i==labels].mean(axis=0) for i in range(k)])
        print(f"new centroids:\n{new_centroids}")
        
        if np.allclose(centroids,new_centroids):
            break

        centroids = new_centroids
    return centroids, labels

data = np.arange(10,dtype=np.float32).reshape((5,2))
k = 3
print(f"{data}")
kmeans(data,k)    
    