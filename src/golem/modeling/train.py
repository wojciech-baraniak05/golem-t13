def train(loader: DataLoader, model: nn.Module, loss_fn: nn.Module , optimizer: torch.optim.Optimizer):
    size: int = len(loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred: torch.Tensor = model(X)
        loss: torch.Tensor = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current : int = (batch + 1) * len(X)
            print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")