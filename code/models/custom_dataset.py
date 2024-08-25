from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, socs=None):
        self.data = data
        self.labels = labels
        if socs is not None: self.socs = socs
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, ind):
        x = self.data[ind]
        y = self.labels[ind]

        if hasattr(self, 'socs'):
            z = self.socs[ind]
            return x, y, z
        else:
            return x, y