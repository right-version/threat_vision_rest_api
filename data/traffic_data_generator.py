import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class TrafficDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: pd.DataFrame with preprocessed features
        """
        self.data = data

        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data[idx])
        tensor.to(torch.float)
        return tensor

