import pandas as pd
import torch as t
import numpy as np


class PacketDataset(t.utils.data.Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = pd.read_csv(data_file)
        self.labels = self.data["label"]
        self.string_to_label_dict = self._generate_string_to_label_dict()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data
        value = data.iloc[idx, :-2]
        attack_category = data["attack_cat"][idx]
        value = value.map(
            lambda x: x
            if x not in self.string_to_label_dict
            else self.string_to_label_dict[x]
        )
        categories_dict = {
            "Normal": 0.0,
            "Fuzzers": 0.0,
            "Analysis": 0.0,
            "Backdoors": 0.0,
            "DoS": 0.0,
            "Exploits": 0.0,
            "Generic": 0.0,
            "Reconnaissance": 0.0,
            "Shellcode": 0.0,
            "Worms": 0.0,
        }
        categories_dict[attack_category] = 1.0
        # categories = np.append(list(categories_dict.values()),[0,0,0,0,0,0,0,0])
        categories = np.array(list(categories_dict.values()))
        categories = categories.reshape((10, 1))
        return value.to_numpy(), categories

    def _generate_string_to_label_dict(self):
        use_cols = ["proto", "state", "service"]
        data = pd.read_csv(self.data_file, usecols=use_cols)
        labels = set(
            data.loc[:, "proto"].values.tolist()
            + data.loc[:, "state"].values.tolist()
            + data.loc[:, "service"].values.tolist()
        )
        return dict(zip(labels, [i for i in range(len(labels))]))
