import pandas as pd
import numpy as np
import os


def make_empty_dataset(min_t, max_t, cols, delta_t):
    timestamps = np.arange(0, max_t - min_t, delta_t)
    empty_dataset = pd.DataFrame(index=timestamps, columns=cols)

    return empty_dataset


def granulize(data, granularity):
    print(data.head())
    min_t = min(data.time)
    max_t = max(data.time)
    cols = data.drop("time", axis = 1).columns

    empty_dataset = make_empty_dataset(min_t, max_t, cols, granularity)
    print("empty dataset: " + str(empty_dataset.shape[0]))
    print(empty_dataset.head())

    for i in range(0, len(empty_dataset.index)):
        relevant_rows = data[
            (data["time"] - min_t >= i * granularity) &
            (data["time"] - min_t < (i + 1) * granularity)
        ]

        if i%100 == 0:
            print(i)

        for col in empty_dataset.columns:
            if len(relevant_rows) > 0:
                empty_dataset.loc[empty_dataset.index[i], col] = np.nanmean(relevant_rows[col])
            else:
                raise ValueError("No relevant rows.")

    return empty_dataset


def load_data(path):
    data = pd.read_csv(path + "merged_data.csv")
    data = data.drop("Unnamed: 0", axis = 1)

    return data


def main():
    path = ""
    granularity = 0.1 # in s

    data = load_data(path)

    granulized_data = granulize(data, granularity)

    print(granulized_data.head())

    granulized_data.to_csv("granulized_data.csv")




if __name__ == '__main__':
    main()