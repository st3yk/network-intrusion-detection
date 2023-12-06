#!/bin/env python3
import numpy as np
from torch.utils.data import DataLoader
from dl_project.PacketDataset import PacketDataset

if __name__ == "__main__":
    training_data = PacketDataset("datasets/UNSW_NB15_training.csv")
    print(training_data.generate_string_to_label_dict())
    # test_data = PacketDataset("datasets/UNSW_NB15_testing.csv")
    # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # for i in range(10):
    #     print(next(iter(train_dataloader)))

    # dh = DataHandler("datasets/UNSW_NB15_testing.csv")
    # data = dh.get_data()
    # features = data.dtype.names
    # print(f"features - {features}")
    # r = range(len(data))
    # test = [[data[j][i] for i in range(len(data[j]))] for j in r]
    # # print(np.array(test))
    # # print(data.shape)
    # # for x in np.array(test):
    # #     print(x)

    # print("Correlation matrix")
    # print()

    # for x in np.corrcoef(np.array(test), rowvar=False):
    #     print(x)
