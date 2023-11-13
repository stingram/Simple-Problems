import numpy as np
from collections import Counter
from typing import Tuple

def gini_impurity(y: np.ndarray) -> float:
    impurity = 1
    hist = np.bincount(y)
    ps = hist /len(y)
    for p in ps:
        impurity -= p**2
    return impurity


class Node:
    def __init__(self, feature=None, threshold=None, left: 'Node' = None, right: 'Node' = None,*,value=None):
        self.feature_idx = feature
        self.threshold = threshold
        self.left: 'Node' = left
        self.right: 'Node' = right
        self.value: int = value
        
    def is_leaf(self):
        return self.value is not None
    

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split: int = min_samples_split
        self.max_depth: int = max_depth
        self.n_features: int = n_features
        self.root: Node = None
        
    def _most_common_label(self, y):
        counter = Counter(y)
        # get most common, returns list of tuples, where it's (value,count)
        # So we use first zero to get first and only tuple
        # Then the second zero to get first element in the tuple
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _split(self, X_column: np.ndarray, split_threshold) -> Tuple[np.ndarray,np.ndarray]:
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs
    
    def _gini_impurity(self, X_column: np.ndarray, y, split_threshold) -> float:
        
        # generate split
        left_idxs, right_idxs = self._split(X_column, split_threshold)
        
        # weighted average child impurity (assume only two classes)
        # 1 - sum_labels((label_i/total_labels)**2)
        total_left = len(left_idxs)
        total_right = len(right_idxs)
        
        # calculate gini_impurity for left leaf and right leaf       
        left_gini = gini_impurity(y[left_idxs])
        right_gini = gini_impurity(y[right_idxs])
                
        # weighted sum
        total_gini = (total_left/(total_left+total_right))*left_gini + (total_right/(total_left+total_right))*right_gini
        
        # return gini impurity
        return total_gini
        
    def _best_criteria(self, X: np.ndarray, y: np.ndarray, feat_idxs: np.ndarray):
        best_impurity = 0.5 # the best impurity is 0
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                impurity = self._gini_impurity(X_column, y, threshold)
                
                if impurity < best_impurity:
                    best_impurity = impurity
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh
    
    # def _best_criteria(self, X, y, feat_idxs):
    #     best_gain = -1
    #     split_idx, split_thresh = None, None
    #     for feat_idx in feat_idxs:
    #         X_column = X[:, feat_idx]
    #         thresholds = np.unique(X_column)
    #         for threshold in thresholds:
    #             gain = self._information_gain(y, X_column, threshold)

    #             if gain > best_gain:
    #                 best_gain = gain
    #                 split_idx = feat_idx
    #                 split_thresh = threshold

    #     return split_idx, split_thresh
        
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # stopping criteria - we have a leaf
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        
        # Now if didn't meet stopping criteria
        
        # Select feature indices
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        
        # Do greedy search
        best_feature, best_threshold = self._best_criteria(X,y,feat_idxs)
        
        # now that we found the best, we split the data and recurse down left and right side
        left_idxs, right_idxs = self._split(X[:,best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs,:],y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:],y[right_idxs], depth+1)
        
        # return the node
        return Node(best_feature,best_threshold,left,right)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        # get number of features
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)       
        self.root = self._grow_tree(X,y)
        
    def _traverse_tree(self, x: np.ndarray, node: Node):
        # check if we have reached leaf
        if node.is_leaf():
            return node.value
        
        # check threshold to see if we go down left or right tree
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x,node.left)
        else:
            return self._traverse_tree(x,node.right)
        
    def predict(self, X: np.ndarray):
        # traverse tree
        return np.asarray([self._traverse_tree(x, self.root) for x in X])
    