import numpy as np
import sklearn.datasets as datasets
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import json
from pathlib import Path


class Pipeline:
    
    def __init__(self, seed: int = 42, n: int = 5000, noise: float = 0.3):
        self.X, self.y = datasets.make_moons(n_samples=n, noise=noise, random_state=seed)
        self.loaders: List[Optional[DataLoader]] = [None, None, None]
        self.X_sa_part: List[Optional[np.ndarray]] = [None, None, None]
        self.y_sa_part: List[Optional[np.ndarray]] = [None, None, None]
        self.embeddings: List[Optional[np.ndarray]] = [None, None, None]
        self.seed = seed
        self.n_samples = n
        self.noise = noise

    def get_data(self, train_ratio: float = 0.6, val_ratio: float = 0.2, \
                 test_ratio: float = 0.2, batch_size: int = 64) -> None:
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=(1 - train_ratio), random_state=self.seed
        )
        val_split = test_ratio / (test_ratio + val_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_split, random_state=self.seed
        )
        
        self.X_sa_part[0] = X_train
        self.y_sa_part[0] = y_train
        self.X_sa_part[1] = X_val
        self.y_sa_part[1] = y_val
        self.X_sa_part[2] = X_test
        self.y_sa_part[2] = y_test

        transform = lambda x, y: TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1)
        )
        loader = lambda x: DataLoader(x, batch_size=batch_size, shuffle=True)

        self.loaders[0] = loader(transform(X_train, y_train))
        self.loaders[1] = loader(transform(X_val, y_val))
        self.loaders[2] = loader(transform(X_test, y_test))

    def save_data(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'X_sa_part': self.X_sa_part,
            'y_sa_part': self.y_sa_part,
            'seed': self.seed,
            'n_samples': self.n_samples,
            'noise': self.noise
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dane zapisane do {filepath}")

    def load_data(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.X_sa_part = data['X_sa_part']
        self.y_sa_part = data['y_sa_part']
        self.seed = data['seed']
        self.n_samples = data['n_samples']
        self.noise = data['noise']
        print(f"Dane zaladowane z {filepath}")