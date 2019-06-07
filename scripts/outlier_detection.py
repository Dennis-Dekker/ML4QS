import pandas as pd
import numpy as np
import math
from scipy import spatial, special
import copy


def distance_table(dataset, cols, d_function):
    dataset = pd.DataFrame(spatial.distance.squareform(distance(dataset.ix[:, cols], d_function)),
                           columns=dataset.index, index=dataset.index)

    return dataset


def distance(rows, d_function='euclidean'):
    if d_function == 'euclidean':
        # Assumes m rows and n columns (attributes), returns and array where each row represents
        # the distances to the other rows (except the own row).
        return spatial.distance.pdist(rows, 'euclidean')
    else:
        raise ValueError("Unknown distance value '" + d_function + "'")


def normalize_dataset(dataset, cols):
    dataset_norm = copy.deepcopy(dataset)
    for col in cols:
        dataset_norm[col] = (dataset[col] - dataset[col].mean() / dataset[col].max() - dataset[col].min())

    return dataset_norm


def simple_distance_based(dataset, cols, d_metric, d_min, f_min):
    # Normalize the dataset first.
    new_dataset = normalize_dataset(dataset.dropna(axis=0, subset=cols), cols)
    # Create the distance table first between all instances:
    distances = distance_table(new_dataset, cols, d_metric)

    mask = []
    # Pass the rows in our table.
    for i in range(0, len(new_dataset.index)):
        # Check what faction of neighbors are beyond dmin.
        frac = (float(sum([1 for col_val in distances.ix[i, :].tolist() if col_val > d_min])) / len(
            new_dataset.index))
        # Mark as an outlier if beyond the minimum frequency.
        mask.append(frac > f_min)
    data_mask = pd.DataFrame(mask, index=new_dataset.index, columns=['simple_dist_outlier'])
    data_table = pd.concat([dataset, data_mask], axis=1)
    return data_table


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
    # Load data
    dataset = load_data("../data/processed_data.csv")

    # Chauvenet
    chauvenet_cols = ["Gain"]

    dataset = apply_chauvenet(dataset, chauvenet_cols)

    # Simple distance based
    distance_cols = ["gFx", "gFy", "gFz"]
    d_metric = "euclidean"
    d_min = 0.10
    f_min = 0.99

    dataset = simple_distance_based(dataset, distance_cols, d_metric, d_min, f_min)

    # Save dataset
    dataset.to_csv("../data/data_chauvenet_gain.csv")


if __name__ == '__main__':
    main()
