import pandas as pd
import numpy as np
import os


def granulize(data, granularity):
    print(data.head())

    for i in range(0, len(data.index)):
        pass


    return data


def load_data(path):
    data = pd.read_csv(path + "merged_data.csv", index_col="time")

    return data


def main():
    path = ""
    granularity = 250 # in ms

    data = load_data(path)

    granulized_data = granulize(data, granularity)




if __name__ == '__main__':
    main()