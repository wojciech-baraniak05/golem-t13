def test(dataloader, model, loss_fn, text='Val'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred_logits = model(X)
            test_loss += loss_fn(pred_logits, y).item()
            predicted_labels = (pred_logits > 0).float()
            correct += (predicted_labels == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    
    print(f"{text} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")