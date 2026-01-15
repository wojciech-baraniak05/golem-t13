import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def test(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, Device: torch.device, text: str ='Val') -> float:
    size: int = len(dataloader.dataset)
    num_batches: int = len(dataloader)
    model.eval()
    test_loss :float = 0.0
    correct: float = 0.0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(Device), y.to(Device)
            pred_logits: torch.Tensor = model(X)
            
            if pred_logits.shape[-1] > 1:
                test_loss += loss_fn(pred_logits, y.long()).item()
                predicted_labels: torch.Tensor = torch.argmax(pred_logits, dim=1)
                correct += (predicted_labels == y.long().squeeze()).sum().item()
            else:
                test_loss += loss_fn(pred_logits, y).item()
                predicted_labels: torch.Tensor = (pred_logits > 0).float()
                correct += (predicted_labels == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    return correct