##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

# Let is create our visualization class again.
DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = '../Data/'
dataset = pd.read_csv(dataset_path + 'assignment3_our_result_outliers.csv', index_col=0)
dataset.index = dataset.index.to_datetime()

# Computer the number of milliseconds covered by an instane based on the first two rows
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

print(dataset.index[1])

# Step 2: Let us impute the missing values.

MisVal = ImputationMissingValues()
imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'hr')
imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'hr')
imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'hr')
DataViz.plot_imputed_values(dataset, ['original', 'mean', 'interpolation'], 'hr', imputed_mean_dataset['hr'],
                            imputed_interpolation_dataset['hr'])

# And we impute for all columns except for the label in the selected way (interpolation)

for col in [c for c in dataset.columns if not 'label' in c]:
    dataset = MisVal.impute_interpolate(dataset, col)

# Let us try the Kalman filter on the light_phone_lux attribute and study the result.

original_dataset = pd.read_csv(dataset_path + 'assignment3_our_result_outliers.csv', index_col=0)
original_dataset.index = original_dataset.index.to_datetime()
KalFilter = KalmanFilters()
kalman_dataset = KalFilter.apply_kalman_filter(original_dataset, 'hr')
DataViz.plot_imputed_values(kalman_dataset, ['original', 'kalman'], 'hr', kalman_dataset['hr_kalman'])
DataViz.plot_dataset(kalman_dataset, ['hr', 'hr_kalman'], ['exact', 'exact'], ['line', 'line'])

# We ignore the Kalman filter output for now...

# Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

LowPass = LowPassFilter()

# Determine the sampling frequency.
fs = float(1000) / milliseconds_per_instance
cutoff = 1.5

# Let us study hr:
new_dataset = LowPass.low_pass_filter(copy.deepcopy(dataset), 'hr', fs, cutoff, order=10)
DataViz.plot_dataset(new_dataset.ix[int(0.4 * len(new_dataset.index)):int(0.43 * len(new_dataset.index)), :],
                     ['hr', 'hr_lowpass'], ['exact', 'exact'], ['line', 'line'])

# And not let us include all measurements that have a form of periodicity (and filter them):
periodic_measurements = ['acc_x', 'acc_y', 'acc_z', 'hr', 'eda', 'temp', 'bvp']

for col in periodic_measurements:
    dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=10)
    dataset[col] = dataset[col + '_lowpass']
    del dataset[col + '_lowpass']

# Determine the PC's for all but our target columns (the labels and the heart rate)
# We simplify by ignoring both, we could also ignore one first, and apply a PC to the remainder.

PCA = PrincipalComponentAnalysis()
selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c)) and (not (c == 'hr'))]
pc_values = PCA.determine_pc_explained_variance(dataset, selected_predictor_cols)

# Plot the variance explained.

plot.plot(range(1, len(selected_predictor_cols) + 1), pc_values, 'b-')
plot.xlabel('principal component number')
plot.ylabel('explained variance')
plot.show(block=False)

# We select 7 as the best number of PC's as this explains most of the variance

n_pcs = 5

dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

# And we visualize the result of the PC's

DataViz.plot_dataset(dataset, ['pca_', 'hr'], ['like', 'like'], ['line', 'line'])

# And the overall final dataset:

print dataset.head()

DataViz.plot_dataset(dataset, ['acc_', 'hr', 'temp', 'eda', 'bvp','pca_'],
                     ['like', 'like', 'like', 'like', 'like', 'like'],
                     ['line', 'line', 'line', 'line', 'line', 'line'])

# Store the outcome.

dataset.to_csv(dataset_path + 'chapter3_our_result_final.csv')
