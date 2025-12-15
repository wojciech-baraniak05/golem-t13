import numpy as np
import sklearn.datasets as datasets
from typing import List, Optional
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
class Pipeline:
    def __init__(self, seed: int = 42, n: int = 5000, noise: float = 0.3):
        self.X, self.y = datasets.make_moons(n_samples=n, noise=noise, random_state=seed)
        self.loaders:   List[Optional[DataLoader]] = [None, None, None]
        self.X_sa_part: List[Optional[np.ndarray]] = [None, None, None]
        self.y_sa_part: List[Optional[np.ndarray]] = [None, None, None]
        self.embedings: List[Optional[np.ndarray]] = [None, None, None]

    def get_data(self, train_ratio : float = 0.6, val_ratio: float = 0.2, \
                 test_ratio: float = 0.2, batch_size: int = 64) -> None:
        
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=(1 - train_ratio), random_state=42)
        val_split = test_ratio / (test_ratio + val_ratio)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_split, random_state=42)
        self.X_sa_part[0] = X_train; self.y_sa_part[0] = y_train
        self.X_sa_part[1] = X_val;   self.y_sa_part[1] = y_val
        self.X_sa_part[2] = X_test;  self.y_sa_part[2] = y_test

        transform = lambda x,y: TensorDataset(torch.tensor(x, dtype=torch.float32), \
                                              torch.tensor(y, dtype=torch.float32).view(-1,1))
        loader = lambda x: DataLoader(x, batch_size=batch_size, shuffle=True)

        self.loaders[0] = loader(transform(X_train, y_train))
        self.loaders[1] = loader(transform(X_val, y_val))
        self.loaders[2] = loader(transform(X_test, y_test))
    


             



            