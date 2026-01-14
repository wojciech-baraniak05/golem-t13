import numpy as np
from collections import Counter
from .MyDecisionTreeClassifier import MyDecisionTreeClassifier

class MyRandomForestClassifier:
    
    def __init__(self, n_estimators=20, min_samples_split=2, max_depth=10, n_features=None):
        self.n_estimators = n_estimators     
        self.min_samples_split = min_samples_split  # Minimalna liczba próbek do podziału węzła
        self.max_depth = max_depth            # Maksymalna głębokość pojedynczego drzewa
        self.n_features = n_features          # Liczba cech brana pod uwagę przy podziale (losowość cech)
        self.trees = []                       # Lista, w której będziemy trzymać wytrenowane drzewa

    def fit(self, X, y):
        # Resetujemy listę drzew i upewniamy się, że dane to tablice numpy
        self.trees = []
        X = np.array(X)
        y = np.array(y).ravel().astype(int)

        # Główna pętla budująca las - tworzymy tyle drzew, ile zdefiniowaliśmy w n_estimators
        for _ in range(self.n_estimators):
            # Tworzymy instancję pojedynczego drzewa decyzyjnego
            tree = MyDecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features
            )
            
            # tworzymy losowy podzbiór danych treningowych dla tego konkretnego drzewa
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Trenujemy to konkretne drzewo na wylosowanym wycinku danych
            tree.fit(X_sample, y_sample)
            
            # Zapisujemy nauczone drzewo do naszej listy
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        # Losujemy indeksy wierszy od 0 do n_samples (ze zwracaniem -> replace=True)
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        # Zwracamy dane (X) i etykiety (y) odpowiadające wylosowanym indeksom
        return X[idxs], y[idxs]

    def predict(self, X):
        X = np.array(X)
        # Zbieramy predykcje od KAŻDEGO drzewa w lesie dla danych wejściowych
        # Wynik to tablica o wymiarach [liczba_drzew, liczba_próbek]
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Transponujemy macierz (.T), żeby teraz wiersze to były próbki, a kolumny to głosy drzew
        # Wynik: [liczba_próbek, liczba_drzew]
        tree_preds = tree_preds.T
        
        # Dla każdej próbki (axis=1) sprawdzamy, która klasa występuje najczęściej (np.bincount + argmax)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=tree_preds)

    def _most_common_label(self, y):
        # Pomocnicza metoda do znalezienia najczęstszego elementu w liście/tablicy
        counter = Counter(y)
        if not counter: 
            return 0
        # Zwraca najczęściej występującą etykietę (klasę)
        return counter.most_common(1)[0][0]