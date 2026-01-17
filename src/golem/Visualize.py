import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_circles, load_digits, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

from models.MyDecisionTreeClassifier import MyDecisionTreeClassifier
from models.MyRandomForestClassifier import MyRandomForestClassifier
from models.MLPClassifier import MLPClassifier

class MLPWrapper:
    """
    Klasa pomocnicza, która nadaje Twojemu modelowi MLPClassifier interfejs 
    zgodny ze Scikit-Learn (metody fit i predict), aby łatwo go wizualizować.
    """
    def __init__(self, depth=4, hidden_dim=32, epochs=200, lr=0.01, device='cpu'):
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device(device)
        self.model = None
        self.out_dim = None

    def fit(self, X, y):
        # Określenie wymiarów wejścia i wyjścia
        input_dim = X.shape[1]
        classes = np.unique(y)
        self.out_dim = len(classes) if len(classes) > 2 else 1
        
        # Inicjalizacja Twojego modelu MLPClassifier
        self.model = MLPClassifier(
            depth=self.depth, 
            input_dim=input_dim, 
            hidden_dim=self.hidden_dim, 
            out_dim=self.out_dim
        ).to(self.device)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        if self.out_dim == 1:
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
            criterion = nn.BCEWithLogitsLoss()
        else:
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
            criterion = nn.CrossEntropyLoss()
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
        if self.out_dim == 1:
            predicted = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy().flatten()
        else:
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
            
        return predicted.astype(int)

class HybridMLPTreeWrapper:
    """
    To jest klasa, która łączy MLP jako ekstraktor cech i Twoje Drzewo jako klasyfikator.
    """
    def __init__(self, mlp_params, tree_params, device='cpu'):
        self.mlp_wrapper = MLPWrapper(**mlp_params, device=device)
        self.tree = MyDecisionTreeClassifier(**tree_params)
        self.device = torch.device(device)

    def fit(self, X, y):
        print("  -> Trenowanie MLP w hybrydzie...")
        self.mlp_wrapper.fit(X, y)
        
        self.mlp_wrapper.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            X_embeddings = self.mlp_wrapper.model(X_tensor, embedding_flag=True).cpu().numpy()
            
        print("  -> Trenowanie Drzewa na embeddingach...")
        self.tree.fit(X_embeddings, y)
        return self

    def predict(self, X):
        self.mlp_wrapper.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            X_embeddings = self.mlp_wrapper.model(X_tensor, embedding_flag=True).cpu().numpy()
            
        return self.tree.predict(X_embeddings)
def plot_decision_boundary(clf, X, y, ax, title):
    """Rysuje granice decyzyjne klasyfikatora."""
    h = .02  # krok siatki
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    try:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    except Exception as e:
        print(f"Błąd predykcji dla {title}: {e}")
        return

    Z = Z.reshape(xx.shape)
    
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF', "#4A0701", '#00FF00' , "#034D03", "#736EE4", "#2C412C", "#800BCA", "#D7FD00", "#00D5FF"][:len(np.unique(y))])
    
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')
    ax.set_title(title)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


datasets = []

X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
datasets.append(("Moons", X_moons, y_moons))

X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
datasets.append(("Circles", X_circles, y_circles))

digits = load_digits()

X_digits, y_digits = digits.data, digits.target

mask = np.isin(y_digits, [0, 1, 2]) 
X_digits = X_digits[mask]
y_digits = y_digits[mask]

pca = PCA(n_components=2)
X_digits_2d = pca.fit_transform(X_digits)


scaler = StandardScaler()
X_digits_2d = scaler.fit_transform(X_digits_2d)
datasets.append(("Digits (PCA 2D)", X_digits_2d, y_digits))

X_blobs, y_blobs = make_blobs(n_samples=300, centers=12, cluster_std=1.5, random_state=42)
datasets.append(("Blobs (6 classes)", X_blobs, y_blobs))

classifiers = [
    ("Decision Tree", MyDecisionTreeClassifier(max_depth=5)),
    ("Random Forest", MyRandomForestClassifier(n_estimators=10, max_depth=5)),
    ("MLP", MLPWrapper(depth=4, hidden_dim=32, epochs=500, lr=0.01)),
    ("Hybrid (MLP+Tree)", HybridMLPTreeWrapper(
        mlp_params={'depth': 4, 'hidden_dim': 64, 'epochs': 400, 'lr': 0.01},
        tree_params={'max_depth': 6}
    ))
]

fig, axes = plt.subplots(len(datasets), len(classifiers), figsize=(15, 12))
noise_val = [0.2, 1, 10]
for i in noise_val:
    X_moons, y_moons = make_moons(n_samples=200, noise=i, random_state=42)
    datasets.append(("Moons", X_moons, y_moons))

    X_circles, y_circles = make_circles(n_samples=200, noise=i, factor=0.5, random_state=42)
    datasets.append(("Circles", X_circles, y_circles))
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=12, cluster_std=1.5, random_state=42)
    datasets.append(("Blobs (6 classes)", X_blobs, y_blobs))
    for i, (ds_name, X, y) in enumerate(datasets):

        print(f"\n--- Zbiór danych: {ds_name} ---")

        for j, (clf_name, clf) in enumerate(classifiers):
            ax = axes[i, j]
            
            # Trenowanie
            try:
                clf.fit(X, y)
                
                y_pred = clf.predict(X)
                acc = accuracy_score(y, y_pred)
                print(f"Model: {clf_name:20} | Accuracy: {acc:.2%}")

                plot_decision_boundary(clf, X, y, ax, f"{clf_name}\nAcc: {acc:.2f}") # Dodano acc też do tytułu wykresu
            except Exception as e:
                print(f"Nie udało się wytrenować {clf_name}: {e}")
                ax.text(0.5, 0.5, "Błąd treningu", ha='center')

plt.tight_layout()
plt.show()
