import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_dataset(dataset):
    plt.plot(dataset.loc[:, ["gFx", "gFy", "gFz"]])
    plt.legend(("gFx", "gFy", "gFz"))
    plt.show()


def make_empty_dataset(min_t, max_t, cols, delta_t):
    timestamps = np.arange(0, max_t - min_t, delta_t)
    empty_dataset = pd.DataFrame(index=timestamps, columns=cols)

    return empty_dataset


def create_dataset(df_raw, delta_t):
    min_t = min(df_raw.time)
    max_t = max(df_raw.time)
    cols = df_raw.drop(["time"], axis=1).columns

    empty_dataset = make_empty_dataset(min_t, max_t, cols, delta_t)

    for i in range(0, len(empty_dataset.index)):
        relevant_rows = df_raw[
            (df_raw["time"] - min_t >= i * delta_t) &
            (df_raw["time"] - min_t < (i + 1) * delta_t)
            ]

        for col in empty_dataset.columns:
            if len(relevant_rows) > 0:
                empty_dataset.loc[empty_dataset.index[i], col] = np.average(relevant_rows[col])
            else:
                raise ValueError("No relevant rows.")

    return empty_dataset


def preprocess(data):
    df_raw = data.drop(["Unnamed: 18"], axis=1)
    df_raw = df_raw[df_raw.Gain != "-∞"]
    df_raw = df_raw.astype("float64")

    return df_raw


def main():
    # Variables
    delta_t = 0.25

    data = pd.read_csv("../data/ML4QS_testrun_1")

    df_raw = preprocess(data)

    dataset = create_dataset(df_raw, delta_t)

    plot_dataset(dataset)


if __name__ == '__main__':
    main()
