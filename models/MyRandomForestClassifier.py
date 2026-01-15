import numpy as np
from typing import Tuple
from collections import Counter
from .MyDecisionTreeClassifier import MyDecisionTreeClassifier


class MyRandomForestClassifier:
    def __init__(self, n_estimators: int = 20, min_samples_split: int = 2, max_depth: int = 10, n_features: int = None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        X = np.array(X)
        y = np.array(y).ravel().astype(int)

        for _ in range(self.n_estimators):
            tree = MyDecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features
            )
            
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X: np.ndarray)-> np.ndarray:
        X = np.array(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = tree_preds.T
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=tree_preds)

    def _most_common_label(self, y: np.ndarray)-> int:
        counter = Counter(y)
        if not counter: 
            return 0
        return int(counter.most_common(1)[0][0])