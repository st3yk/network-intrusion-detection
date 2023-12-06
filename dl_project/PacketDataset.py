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

    def generate_string_to_label_dict(self):
        use_cols = ["proto", "state", "service"]
        data = pd.read_csv(self.data_file, usecols=use_cols)
        labels = set(data[0] + data[1] + data[2])
        label_map = dict(zip(labels, [i for i in range(len(labels))]))

        return label_map
