
def prepare_data(X, y, multiclass=False, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), random_state=42)
    val_split = test_ratio / (test_ratio + val_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_split, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)

    if multiclass:
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val   = torch.tensor(y_val, dtype=torch.long)
        y_test  = torch.tensor(y_test, dtype=torch.long)
    else:
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_val   = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        y_test  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)
    test_dataset  = TensorDataset(X_test, y_test)
    return train_dataset, val_dataset, test_dataset

def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
