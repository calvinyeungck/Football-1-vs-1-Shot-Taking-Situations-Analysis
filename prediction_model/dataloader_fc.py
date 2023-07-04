import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data.to_numpy())
        self.targets = torch.tensor(targets.to_numpy())
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y