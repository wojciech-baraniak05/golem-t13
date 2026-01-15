import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def train(loader: DataLoader, model: nn.Module, loss_fn: nn.Module , optimizer: torch.optim.Optimizer, Device: torch.device):
    size: int = len(loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(Device), y.to(Device)

        pred: torch.Tensor = model(X)
        loss: torch.Tensor = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()