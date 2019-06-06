import pandas as pd
import numpy as np
import math
from scipy import special


def apply_chauvenet(dataset, col):
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (2 * N)

    deviation = abs(dataset[col] - mean) / std

    low = -deviation / math.sqrt(2)
    high = deviation / math.sqrt(2)
    prob = []
    mask = []

    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(1.0 - 0.5 * (special.erf(high.iloc[i]) - special.erf(low.iloc[i])))
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col[0] + '_outlier'] = mask

    return dataset


def load_data(file_name):
    dataset = pd.read_csv(file_name)

    return dataset


def main():
    dataset = load_data("../data/processed_data.csv")

    chauvenet_cols = ["Gain"]

    dataset = apply_chauvenet(dataset, chauvenet_cols)

    print(dataset)

    dataset.to_csv("../data/data_chauvenet_gain.csv")


if __name__ == '__main__':
    main()
