import os
import pandas as pd
import torch as t


class PacketDataset(t.utils.data.Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = pd.read_csv(data_file)
        self.labels = self.data["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data
        value = data.iloc[idx, :-2]
        attack_category = data["attack_cat"][idx]
        print(value)
        print(attack_category)
        return value.to_numpy(), attack_category
