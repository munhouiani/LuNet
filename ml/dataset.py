import numpy as np
from torch.utils.data import Dataset


class UNSWNB15Dataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        attack_cat = self.data[idx, -2]
        label = self.data[idx, -1]
        feature = self.data[idx, 0:-2]

        return {
            'feature': feature,
            'attack_cat': attack_cat,
            'label': label
        }
