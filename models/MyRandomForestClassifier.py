import numpy as np
from collections import Counter
# Importujemy Twoje istniejące drzewo
from models.MyDecisionTreeClassifier import MyDecisionTreeClassifier

class MyRandomForestClassifier:
    def __init__(self, n_estimators=20, min_samples_split=2, max_depth=10, n_features=None):
        self.n_estimators = n_estimators          # Liczba drzew
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        X = np.array(X)
        y = np.array(y).ravel() # Spłaszczamy y (N,)
        y = y.astype(int)

        for _ in range(self.n_estimators):
            # Tworzymy nowe drzewo
            tree = MyDecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features
            )
            
            # --- BOOTSTRAP (Losowanie ze zwracaniem) ---
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Trenujemy drzewo na próbce
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        # Losujemy indeksy (mogą się powtarzać)
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        X = np.array(X)
        # Zbieramy głosy od każdego drzewa
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # Zamiana osi: (n_drzew, n_próbek) -> (n_próbek, n_drzew)
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        # Głosowanie większościowe
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        if not counter: return 0
        return counter.most_common(1)[0][0]