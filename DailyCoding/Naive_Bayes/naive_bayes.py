from typing import Dict, List
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class GaussianNB:
    def __init__(self):
        self.priors:Dict[int,float] = {}
        self.likelihood_means = np.array([])
        self.likelihood_var = np.array([])
        self.n_features = 1
        self.labels = []
        return
    
    def fit(self,X: np.ndarray,y: np.ndarray):
        n_samples, n_features = X.shape
        
        # Get self.labels from y
        self.labels = np.unique(y)
                  
        # calculate number of features
        self.n_features = len(X[0])
        
        #  calculate likelihoods for each feature
        self.likelihood_means = np.zeros((len(self.labels),self.n_features),dtype=np.float64)
        self.likelihood_var = np.zeros((len(self.labels),self.n_features),dtype=np.float64)
        
        # assuming label start at 0
        for label in self.labels:

            # get all samples associated with this label
            X_c = X[label==y]
            
            # we'll have a mean and standard deviation for each feature for each class
            self.likelihood_means[label,:] = X_c.mean(axis=0)
            self.likelihood_var[label,:] = X_c.var(axis=0)
            self.priors[label] = X_c.shape[0] / float(n_samples)
        
        return


    def pdf(self, x:np.ndarray,label:int) -> float:
        # given single feature vector compute likelihoods for single class
        mean = self.likelihood_means[label] # 1 x n_features 
        var = self.likelihood_var[label] # 1 x n_features
        
        numerator = np.exp(-(x-mean)**2/(2*var)) # 1 x n_features
        denominator = np.sqrt(2*np.pi*var) # 1 x n_features
        return numerator/denominator # 1 x n_features
        
        
    def _predict(self,x:np.ndarray) -> int:
        posteriors = []
        for idx, c in enumerate(self.priors.keys()):
            prior = np.log(self.priors[idx]) # VERY IMPORTANT TO TAKE THE LOG
            class_conditional = np.sum(np.log(self.pdf(x, c))) # VERY IMPORTANT TO TAKE THE LOG
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self.labels[np.argmax(posteriors)]
    
    def predict(self,X: np.ndarray) -> List[int]:
        return [self._predict(x) for x in X]
    

def accuracy(y_pred:np.ndarray, y_true:np.ndarray) -> float:
    accuracy = np.sum(y_pred == y_true) /len(y_true)
    return accuracy

X, y = make_classification(
    n_features=10,
    n_classes=2,
    n_samples=1000,
    random_state=123,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = GaussianNB()
model.fit(X_train,y_train)

y = model.predict(X_test)
print(f"Accuracy: {accuracy(y,y_test)}.")