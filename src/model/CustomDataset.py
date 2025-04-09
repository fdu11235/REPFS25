import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataframe, device=torch.device("cpu")):
        self.data = dataframe.to_numpy()
        self.data_tensor = torch.tensor(self.data, dtype=torch.float)
        self.label_tensor = torch.tensor(self.data[:, -1], dtype=torch.long)

        self.data_tensor = self.data_tensor.to(device)
        self.label_tensor = self.label_tensor.to(device)

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        features = self.data_tensor[idx, :-1]
        label = self.label_tensor[idx]
        return features, label
