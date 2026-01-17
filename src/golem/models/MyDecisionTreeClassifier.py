from __future__ import annotations 
import numpy as np
from typing import Tuple, Optional

class Node:
    def __init__(self, feature:int = None, threshold:float = None, left: Node = None, right: Node = None, *, value: int = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None

class MyDecisionTreeClassifier:
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, n_features: int = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.array(X)
        y = np.array(y).ravel().astype(int)

        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return Node(value=self._most_common_label(y))

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)
   
    def _best_split(self, X: np.ndarray, y: np.ndarray , feat_idxs: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs: 
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds: 
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        return split_idx, split_thresh
   
    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, threshold: float) -> float:
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh) -> Tuple[np.ndarray, np.ndarray]:
        left_mask = X_column <= split_thresh
        left_idxs = np.where(left_mask)[0]
        right_idxs = np.where(~left_mask)[0]
        return left_idxs, right_idxs
    
    def _entropy(self, y: np.ndarray) -> float:  
        hist = np.bincount(y)
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))

    def _most_common_label(self, y: np.ndarray) -> int:
        if len(y) == 0:
            return 0
        return np.bincount(y).argmax()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)