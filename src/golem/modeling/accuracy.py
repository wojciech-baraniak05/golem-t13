import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def accuracy_score_MLP(dataloader: DataLoader, model: nn.Module, Device: torch.device) -> float:
    size: int = len(dataloader.dataset)
    num_batches: int = len(dataloader)
    model.eval()
    correct: float = 0.0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(Device), y.to(Device)
            pred_logits: torch.Tensor = model(X)
            predicted_labels: torch.Tensor = (pred_logits > 0).float()
            correct += (predicted_labels == y).type(torch.float).sum().item()
            
    correct /= size
    return correct