import numpy as np

class PCA:
    def __init__(self, k: int):
        self.k: int = k
        self.components: np.ndarray = None
        self.mean = 0
        
    def fit(self, X: np.ndarray):
        
        # get mean
        self.mean = np.mean(X,axis=0)
        
        # shift data
        X = X - self.mean
        
        # compute covariance matrix
        covX = np.cov(X.T)
        
        # compute eigenvectors and eigenvalues (each column is an eigenvector)
        eigenvalues, eigenvectors = np.linalg.eig(covX)
        
        # get indices of eigenvalues ordered from largest to smallest
        top_idx = np.argsort(eigenvalues)[::-1]
        
        # reorder eigenvectors (use transpose so we can use indices)
        # so each row is an eigenvector
        eigenvectors = eigenvectors.T[top_idx]
        
        # select top eigenvectors (each row is an eigenvector)
        self.components = eigenvectors[:self.k]
        
        return
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X - self.mean
        # to project a point onto components do np.dot(X,components.T)
        transformed_X = np.dot(X,self.components.T)
        return transformed_X