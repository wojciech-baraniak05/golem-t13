import numpy as np
import sklearn.datasets as datasets
from typing import List, Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from golem.models.MLPClassifier import MLPClassifier
from golem.modeling.train import train
from golem.modeling.test import test
from golem.modeling.accuracy import accuracy_score_MLP


class Pipeline:
    def __init__(self, X: np.ndarray, y: np.ndarray, multiclass: bool = False):
        self.X = X
        self.y = y
        self.multiclass = multiclass
        self.loaders: List[Optional[DataLoader]] = [None, None, None]
        self.X_sa_part: List[Optional[np.ndarray]] = [None, None, None]
        self.y_sa_part: List[Optional[np.ndarray]] = [None, None, None]
        self.embedings: List[Optional[np.ndarray]] = [None, None, None]
        
        self.mlp_model: Optional[MLPClassifier] = None
        self.rf_hybrid_model: Optional[RandomForestClassifier] = None
        self.rf_simple_model: Optional[RandomForestClassifier] = None
        self.dt_hybrid_model: Optional[DecisionTreeClassifier] = None
        self.dt_simple_model: Optional[DecisionTreeClassifier] = None
        self.gb_hybrid_model: Optional[GradientBoostingClassifier] = None
        self.gb_simple_model: Optional[GradientBoostingClassifier] = None

    def get_data(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2, batch_size: int = 64, multiclass: Optional[bool] = None) -> None:
        if multiclass is None:
            multiclass = self.multiclass
            
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=(1 - train_ratio), random_state=42)
        val_split = test_ratio / (test_ratio + val_ratio)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_split, random_state=42)
        self.X_sa_part[0] = X_train
        self.y_sa_part[0] = y_train
        self.X_sa_part[1] = X_val
        self.y_sa_part[1] = y_val
        self.X_sa_part[2] = X_test
        self.y_sa_part[2] = y_test

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        
        if multiclass:
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            y_val_t = torch.tensor(y_val, dtype=torch.long)
            y_test_t = torch.tensor(y_test, dtype=torch.long)
        else:
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        self.loaders[0] = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        self.loaders[1] = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
        self.loaders[2] = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    
    def _get_device(self, device: Optional[torch.device] = None) -> torch.device:
        if device is not None:
            return device
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def fit_mlp(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, multiclass: Optional[bool] = None, epochs: int = 50, depth: int = 7, hidden_dim: int = 64, out_dim: Optional[int] = None, learning_rate: float = 1e-3, batch_size: int = 64, device: Optional[torch.device] = None) -> Tuple[MLPClassifier, float]:
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        
        if multiclass is None:
            multiclass = self.multiclass
            
        device = self._get_device(device)
        
        # Automatically determine output dimension
        if out_dim is None:
            if multiclass:
                out_dim = len(np.unique(y))
            else:
                out_dim = 1
        
        # If out_dim > 1, this must be multiclass classification
        if out_dim > 1:
            multiclass = True
        
        # Prepare data with the correct multiclass setting
        if self.loaders[0] is None:
            self.get_data(batch_size=batch_size, multiclass=multiclass)
        
        input_dim = X.shape[1]
        self.mlp_model = MLPClassifier(depth=depth, input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        self.mlp_model.to(device)
        
        # Use appropriate loss function based on multiclass
        if multiclass:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.BCEWithLogitsLoss()
            
        optimizer = torch.optim.AdamW(self.mlp_model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train(self.loaders[0], self.mlp_model, loss_fn, optimizer, device)
            val_acc = test(self.loaders[1], self.mlp_model, loss_fn, device)
            print(f"Accuracy: {val_acc}")
        
        final_acc = accuracy_score_MLP(self.loaders[2], self.mlp_model, device)
        return self.mlp_model, final_acc
    
    def extract_embeddings(self, model: Optional[MLPClassifier] = None, device: Optional[torch.device] = None) -> None:
        if model is None:
            model = self.mlp_model
        if model is None:
            raise ValueError("No model provided and no mlp_model stored")
            
        device = self._get_device(device)
        
        for idx in range(3):
            if self.loaders[idx] is not None:
                embeddings, labels = model.extract(self.loaders[idx], device)
                self.embedings[idx] = embeddings
    
    def train_random_forest(self, use_embeddings: bool = False, params: Optional[Dict[str, Any]] = None) -> Tuple[RandomForestClassifier, float]:
        if params is None:
            params = {}
        
        if use_embeddings:
            if self.embedings[0] is None or self.embedings[1] is None:
                raise ValueError("Embeddings not extracted")
            X_train, X_val = self.embedings[0], self.embedings[1]
            y_train, y_val = self.y_sa_part[0], self.y_sa_part[1]
        else:
            X_train, X_val = self.X_sa_part[0], self.X_sa_part[1]
            y_train, y_val = self.y_sa_part[0], self.y_sa_part[1]
        
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        acc_val = accuracy_score(y_val, model.predict(X_val))
        
        if use_embeddings:
            self.rf_hybrid_model = model
        else:
            self.rf_simple_model = model
        
        return model, acc_val
    
    def train_decision_tree(self, use_embeddings: bool = False, params: Optional[Dict[str, Any]] = None) -> Tuple[DecisionTreeClassifier, float]:
        if params is None:
            params = {}
        
        if use_embeddings:
            if self.embedings[0] is None or self.embedings[1] is None:
                raise ValueError("Embeddings not extracted")
            X_train, X_val = self.embedings[0], self.embedings[1]
            y_train, y_val = self.y_sa_part[0], self.y_sa_part[1]
        else:
            X_train, X_val = self.X_sa_part[0], self.X_sa_part[1]
            y_train, y_val = self.y_sa_part[0], self.y_sa_part[1]
        
        model = DecisionTreeClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        acc_val = accuracy_score(y_val, model.predict(X_val))
        
        if use_embeddings:
            self.dt_hybrid_model = model
        else:
            self.dt_simple_model = model
        
        return model, acc_val
    
    def train_gradient_boosting(self, use_embeddings: bool = False, params: Optional[Dict[str, Any]] = None) -> Tuple[GradientBoostingClassifier, float]:
        if params is None:
            params = {}
        
        if use_embeddings:
            if self.embedings[0] is None or self.embedings[1] is None:
                raise ValueError("Embeddings not extracted")
            X_train, X_val = self.embedings[0], self.embedings[1]
            y_train, y_val = self.y_sa_part[0], self.y_sa_part[1]
        else:
            X_train, X_val = self.X_sa_part[0], self.X_sa_part[1]
            y_train, y_val = self.y_sa_part[0], self.y_sa_part[1]
        
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        acc_val = accuracy_score(y_val, model.predict(X_val))
        
        if use_embeddings:
            self.gb_hybrid_model = model
        else:
            self.gb_simple_model = model
        
        return model, acc_val
    


             



            