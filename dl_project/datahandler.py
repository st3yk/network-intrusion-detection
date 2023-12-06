import numpy as np
import torch as t
import PacketDataset
import os
import matplotlib.pyplot as plt


#dataset_dir = os.environ["DATASET_DIR"]
# training_set = PacketDataset.PacketDataset(f"{dataset_dir}/UNSW_NB15_training.csv")

# batch_size = 4

# trainloader = t.utils.data.DataLoader(
#     training_set, batch_size=batch_size, shuffle=True, num_workers=2
# )
class DataHandler:
    def __init__(self, path: str, rowCount: int = -1) -> None:
        self._data = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            encoding="utf-8",
            usecols=tuple([1] + [x for x in range(5, 43)]),
        )
        self._name = self._get_name(path)

    def get_data(self):
        return self._data

    def _get_name(self, path):
        if "test" in path:
            return "Testing Dataset"
        elif "train" in path:
            return "Training Dataset"
        else:
            return "Dataset"

    def get_attacks_statistics(self, plot=False) -> None:
        attacks = np.unique(self._data["attack_cat"])
        print(f"Getting statistics on attacks: {attacks}")
        ds_size = len(self._data)
        prct_total = 0
        occurences = dict()
        for attack in attacks:
            number = len([x for x in self._data if x["attack_cat"] == attack])
            occurences[attack] = number
            prct = number * 100 / ds_size
            prct_total += prct
            print(f" -> {attack} - {number}, {prct:.2f}%")
        print(
            f"Total attacks {sum(occurences.values())}, \
                dataset size - {ds_size}, {prct_total}%"
        )
        occurences = dict(sorted(occurences.items(), key=lambda x: x[1]))
        if plot:
            plt.bar(
                occurences.keys(),
                occurences.values(),
                facecolor="#2ab0ff",
                edgecolor="#169acf",
            )
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.title(self._name)
            plt.xlabel("Attack category")
            plt.ylabel("Values")
            plt.show()
