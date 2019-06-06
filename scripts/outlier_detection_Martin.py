import scipy
import math
from sklearn import mixture
import numpy as np
import pandas as pd

# Fits a mixture model towards the data expressed in col and adds a column with the probability
# of observing the value given the mixture model.
def apply_mixture_model(data_table, col):
    # Fit a mixture model to our data.
    data = data_table[data_table[col].notnull()][col]
    g = mixture.GMM(n_components=3, n_iter=1)

    g.fit(data.as_matrix().reshape(-1,1))

    # Predict the probabilities
    probs = g.score(data.as_matrix().reshape(-1,1))

    # Create the right data frame and concatenate the two.
    data_probs = pd.DataFrame(np.power(10, probs), index=data.index, columns=[col[0] + "_mixture"])
    data_table = pd.concat([data_table, data_probs], axis=1)

    return data_table


def load_data(file_name):
    dataset = pd.read_csv(file_name)
    return dataset


def main():
    dataset = load_data("../data/processed_data.csv")

    # chauvenet_cols = ["Gain"]
    mixture_cols = ["Gain"]
    # dataset = apply_chauvenet(dataset, chauvenet_cols)
    dataset = apply_mixture_model(dataset, mixture_cols)

    print(dataset)

    dataset.to_csv('../data/mixture_model.csv')


if __name__ == '__main__':
    main()
