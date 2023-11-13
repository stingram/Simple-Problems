import numpy as np

class LDA:
    def __init__(self, n_components: int):
        self.num_components = n_components
        self.linear_discriminants = None
        
    def fit(self, X, y):
        n_features = X.shape[1] #150 ,4 
        class_labels = np.unique(y)
        
        # Compute S_W, S_B
        
        # need to calculate mean of all samples
        mean_overall = np.mean(X,axis=0)
        
        # create scatter matrices
        S_W = np.zeros((n_features, n_features)) # 4, 4
        S_B = np.zeros((n_features, n_features)) # 4, 4
        
        
        for c in class_labels:
            
            # only select samples for this class
            X_c = X[y==c]
            
            # get in each column -> (1,4), each row is a sample
            mean_c = np.mean(X_c,axis=0)
            
            # n_c, 4 -> 4, n_c. this should be 4,4 matrix
            # want (4, n_c) * (n_c, 4) = 4x4
            #  compute within class scatter
            S_W += (X_c - mean_c).T.dot(X_c-mean_c)
            
            # get number of samples
            n_c = X_c.shape[0]
            
            # get mean diff
            mean_diff = (mean_c - mean_overall).reshape((n_features,1)) # need to have (4,1) not (4,)
            
            # compute between-class scatter
            S_B += n_c*(mean_diff).dot(mean_diff.T)
            
            # compute A matrix
            A = np.linalg.inv(S_W).dot(S_B)
            
            # get eigen values ans eigen vectors of product 
            eigenvalues, eigenvectors = np.linalg.eig(A)
            eigenvectors = eigenvectors.T
            
            # get indices of eigenvalues sorted from largest to smalled
            idxs = np.argsort(abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idxs]
            eigenvectors = eigenvectors[idxs]
            
            self.linear_discriminants = eigenvectors[0:self.num_components]
                      
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        # project data
        return np.dot(X, self.linear_discriminants.T)
    