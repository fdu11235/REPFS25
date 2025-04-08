import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.to_numpy()
        self.data_cuda = torch.tensor(self.data, dtype=torch.float).cuda()
        self.labels_cuda = torch.tensor(self.data[:, -1], dtype=torch.long).cuda()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data_cuda[idx, :-1]
        label = self.labels_cuda[idx]
        return features, label
