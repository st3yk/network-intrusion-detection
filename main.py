#!/bin/env python3
import numpy as np

if __name__ == "__main__":
    dh = DataHandler("datasets/UNSW_NB15_testing.csv")
    data = dh.get_data()
    features = data.dtype.names
    print(f"features - {features}")
    r = range(len(data))
    test = [[data[j][i] for i in range(len(data[j]))] for j in r]
    # print(np.array(test))
    # print(data.shape)
    # for x in np.array(test):
    #     print(x)

    print("Correlation matrix")
    print()

    for x in np.corrcoef(np.array(test), rowvar=False):
        print(x)
