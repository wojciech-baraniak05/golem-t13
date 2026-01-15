import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_embeddings_with_flag(model: nn.Module, loader: DataLoader, device: torch.device):
    
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            embeddings = model(x, embedding_flag=True)     
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
    return np.concatenate(all_embeddings), np.concatenate(all_labels).ravel()


def visualize_embedding_tsne(loader: DataLoader, model: nn.Module, device: torch.device):

    embeddings, labels = get_embeddings_with_flag(model, loader, device)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class Label')
    plt.title('MLP Embedding Space Visualization (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()


def visualize_embedding_pca(loader: DataLoader, model: nn.Module, device: torch.device):

    embeddings, labels = get_embeddings_with_flag(model, loader, device)
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class Label')
    plt.title('MLP Embedding Space Visualization (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()