import pandas as pd
import torch as t
import numpy as np

ATTACK_CAT_TO_ID = {
    "Normal": 0,
    "Reconnaissance": 1,
    "Backdoor": 2,
    "DoS": 3,
    "Exploits": 4,
    "Analysis": 5,
    "Fuzzers": 6,
    "Worms": 7,
    "Shellcode": 8,
    "Generic": 9,
}


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
        attack_category = (
            data["attack_cat"]
            .apply(func=(lambda x: ATTACK_CAT_TO_ID.get(x)))
            .to_numpy()
            .reshape(-1, 1)[idx]
        )
        feature = data.iloc[idx, :-2]
        feature = feature.map(
            lambda x: x
            if x not in self.string_to_label_dict
            else self.string_to_label_dict[x]
        )
        feature = np.expand_dims(feature, axis=0)
        return {
            "feature": feature,
            "attack_category": attack_category,
        }

    def _generate_string_to_label_dict(self):
        use_cols = ["proto", "state", "service"]
        data = pd.read_csv(self.data_file, usecols=use_cols)
        labels = set(
            data.loc[:, "proto"].values.tolist()
            + data.loc[:, "state"].values.tolist()
            + data.loc[:, "service"].values.tolist()
        )
        return dict(zip(labels, [i for i in range(len(labels))]))
