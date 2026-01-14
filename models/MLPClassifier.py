import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader



class MLPClassifier(nn.Module):
    def __init__(self, depth: int = 7, input_dim: int = 2, hidden_dim: int = 64, out_dim: int = 1):
        super().__init__()
        
        if depth < 2:
            raise ValueError("DEPTH < 2")
        
        last_hidden_dim: int = max(hidden_dim // 2, out_dim * 2) 
        hidden_sizes: list[int] = np.linspace(
            start=hidden_dim, 
            stop=last_hidden_dim, 
            num=depth - 1, 
            dtype=int
        ).tolist()
        layer_dims: list[int] = [input_dim] + hidden_sizes

        layers : list[nn.Module] = []
        layers.append(nn.Flatten())
        
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(nn.ReLU())
            
        self.seq: nn.Sequential = nn.Sequential(*layers)
        
        self.head: nn.Linear = nn.Linear(layer_dims[-1], out_dim)
    
    def forward(self, x: torch.Tensor, embedding_flag: bool = False) -> torch.Tensor:
        features: torch.Tensor = self.seq(x)
        if embedding_flag:
            return features
        return self.head(features)
    
    def extract(self, Loader:DataLoader, Device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
        embeddings: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        
        self.eval() 
        with torch.no_grad():
            for X, y in Loader:
                X = X.to(Device)
                emb: torch.Tensor = self.forward(X, embeddingFlag=True)
                embeddings.append(emb.cpu())
                labels.append(y.cpu())
        embeddings: np.ndarray = torch.cat(embeddings, dim=0).numpy()
        labels:np. ndarray = torch.cat(labels, dim=0).numpy()
        return embeddings, labels