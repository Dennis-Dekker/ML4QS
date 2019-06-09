import pandas as pd
import numpy as np
import math
import copy

from scipy import spatial, special
from sklearn import mixture


def reachability_distance(distances, k, i1, i2):
    # Compute the k-distance of i2.
    k_distance_value, neighbors = k_distance(distances, i2, k)
    # The value is the max of the k-distance of i2 and the real distance.
    return max([k_distance_value, distances.ix[i1, i2]])


def local_reachability_density(distances, i, k, k_distance_i, neighbors_i):
    # Set distances to neighbors to 0.
    reachability_distances_array = [0] * len(neighbors_i)

    # Compute the reachability distance between i and all neighbors.
    for i, neighbor in enumerate(neighbors_i):
        reachability_distances_array[i] = reachability_distance(distances, k, i, neighbor)
    if not any(reachability_distances_array):
        return float("inf")
    else:
        # Return the number of neighbors divided by the sum of the reachability distances.
        return len(neighbors_i) / sum(reachability_distances_array)


def k_distance(distances, i, k):
    # Simply look up the values in the distance table, select the min_pts^th lowest value and take the value pairs
    # Take min_pts + 1 as we also have the instance itself in there.
    neighbors = np.argpartition(np.array(distances.ix[i, :]), k + 1)[0:(k + 1)].tolist()
    if i in neighbors:
        neighbors.remove(i)
    return max(distances.ix[i, neighbors]), neighbors


def local_outlier_factor_instance(distances, i, k):
    # Compute the k-distance for i.
    k_distance_value, neighbors = k_distance(distances, i, k)
    # Computer the local reachability given the found k-distance and neighbors.
    instance_lrd = local_reachability_density(distances, i, k, k_distance_value, neighbors)
    lrd_ratios_array = [0] * len(neighbors)

    # Computer the k-distances and local reachability density of the neighbors
    for i, neighbor in enumerate(neighbors):
        k_distance_value_neighbor, neighbors_neighbor = k_distance(distances, neighbor, k)
        neighbor_lrd = local_reachability_density(distances, neighbor, k, k_distance_value_neighbor, neighbors_neighbor)
        # Store the ratio between the neighbor and the row i.
        lrd_ratios_array[i] = neighbor_lrd / instance_lrd

    # Return the average ratio.
    return sum(lrd_ratios_array) / len(neighbors)


def local_outlier_factor(dataset, cols, d_function, k):
    # Inspired on https://github.com/damjankuznar/pylof/blob/master/lof.py
    # but tailored towards the distance metrics and data structures used here.

    # Normalize the dataset first.
    new_dataset = normalize_dataset(dataset.dropna(axis=0, subset=cols), cols)
    # Create the distance table first between all instances:
    distances = distance_table(new_dataset, cols, d_function)

    outlier_factor = []
    # Compute the outlier score per row.
    for i in range(0, len(new_dataset.index)):
        outlier_factor.append(local_outlier_factor_instance(distances, i, k))
    data_outlier_probs = pd.DataFrame(outlier_factor, index=new_dataset.index, columns=['lof'])
    dataset = pd.concat([dataset, data_outlier_probs], axis=1)
    return dataset


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


def apply_mixture_model(data_table, col, c):
    # Fit a mixture model to our data.
    data = data_table[data_table[col].notnull()][col]
    g = mixture.GaussianMixture(n_components=3, n_init=1)

    g.fit(data.as_matrix().reshape(-1, 1))

    # Predict the probabilities
    probs = g.score_samples(data.as_matrix().reshape(-1, 1))

    mask = []
    p_threshold = 1/(c*len(data_table.index))
    for i in probs:
        if np.power(10, i) <= p_threshold:
            mask.append(True)
        else:
            mask.append(False)

    # Create the right data frame and concatenate the two.
    data_mask = pd.DataFrame(mask, index=data_table.index, columns=['Mixture_model'])

    #data_probs = pd.DataFrame(np.power(10, probs), index=data.index, columns=[col[0] + "_mixture"])
    data_table = pd.concat([data_table, data_mask], axis=1)

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
        prob.append(float(1.0 - 0.5 * (special.erf(high.iloc[i]) - special.erf(low.iloc[i]))))
        # And mark as an outlier when the probability is below our criterion.
        if prob[i] <= criterion:
            mask.append(True)
        else:
            mask.append(False)
    dataset["Chauvenet"] = mask

    return dataset


def load_data(file_name):
    dataset = pd.read_csv(file_name)

    return dataset


def main():
    # Load data
    dataset = load_data("../data/processed_data.csv")

    outlier_col = ["p"]

    # Chauvenet
    chauvenet_cols = outlier_col

    dataset = apply_chauvenet(dataset, chauvenet_cols)

    # Mixture model
    mixture_cols = outlier_col
    c = 5

    dataset = apply_mixture_model(dataset, mixture_cols, c)

    # Simple distance based
    distance_cols = outlier_col
    d_metric = "euclidean"
    d_min = 0.10
    f_min = 0.99

    dataset = simple_distance_based(dataset, distance_cols, d_metric, d_min, f_min)

    # local outlier factor
    local_outlier_factor_cols = outlier_col
    d_metric = "euclidean"
    k = 10

    dataset = local_outlier_factor(dataset, local_outlier_factor_cols, d_metric, k)

    # Save dataset
    dataset.to_csv("../data/data_chauvenet_gain.csv")


if __name__ == '__main__':
    main()
