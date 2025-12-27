import torch.nn as nn
import numpy as np
import torch

class MLPClassifier(nn.Module):
    def __init__(self, depth=7, input_dim=2, hidden_dim=64, out_dim=1):
        super().__init__()
        
        if depth < 2:
            raise ValueError("DEPTH < 2")
        
        last_hidden_dim = max(hidden_dim // 2, out_dim * 2) 
        hidden_sizes = np.linspace(
            start=hidden_dim, 
            stop=last_hidden_dim, 
            num=depth - 1, 
            dtype=int
        ).tolist()
        layer_dims = [input_dim] + hidden_sizes

        layers = []
        layers.append(nn.Flatten())
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(nn.ReLU())
            
        self.seq = nn.Sequential(*layers)
        
        self.head = nn.Linear(layer_dims[-1], out_dim)
    
    def forward(self, x, embeddingFlag=False):
        features = self.seq(x)
        if embeddingFlag:
            return features
        return self.head(features)
    
    def extract(self, Loader):
        embeddings = []
        labels = []
        
        self.eval() 
        with torch.no_grad():
            for X, y in Loader:
                X = X.to(DEVICE)
                emb = self.forward(X, embeddingFlag=True)
                embeddings.append(emb.cpu())
                labels.append(y.cpu())
        embeddings = torch.cat(embeddings, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        return embeddings, labels