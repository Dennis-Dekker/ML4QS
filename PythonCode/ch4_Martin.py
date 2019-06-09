from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

import scipy
import math
import numpy as np
import pandas as pd
import copy

# Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
# the sampling rate expresses
# the number of samples per second (i.e. Frequency is Hertz of the dataset).
def find_fft_transformation(data, sampling_rate):
        # Create the transformation, this includes the amplitudes of both the real
        # and imaginary part.
        transformation = np.fft.rfft(data, len(data))
        return transformation.real, transformation.imag

# Get frequencies over a certain window.
def abstract_frequency(data_table, cols, window_size, sampling_rate):

# Create new columns for the frequency data.
    freqs = np.fft.rfftfreq(int(window_size)) *sampling_rate

    for col in cols:
        data_table[col + '_max_freq'] = np.nan
        data_table[col + '_freq_weighted'] = np.nan
        data_table[col + '_pse'] = np.nan
        for freq in freqs:
            data_table[col + '_freq_' + str(freq) + '_Hz_ws_' + str(window_size)] = np.nan

# Pass over the dataset (we cannot compute it when we do not have enough history)
# and compute the values.
    for i in range(window_size, len(data_table.index)):
        for col in cols:
            real_ampl, imag_ampl = self.find_fft_transformation(data_table[col][i-window_size:min(i+1, len(data_table.index))], sampling_rate)
            # We only look at the real part in this implementation.
            for j in range(0, len(freqs)):
                data_table.ix[i, col + '_freq_' + str(freqs[j]) + '_Hz_ws_' + str(window_size)] = real_ampl[j]
            # And select the dominant frequency. We only consider the positive frequencies for now.

            data_table.ix[i, col + '_max_freq'] = freqs[np.argmax(real_ampl[0:len(real_ampl)])]
            data_table.ix[i, col + '_freq_weighted'] = float(np.sum(freqs * real_ampl)) / np.sum(real_ampl)
            PSD = np.divide(np.square(real_ampl),float(len(real_ampl)))
            PSD_pdf = np.divide(PSD, np.sum(PSD))
            data_table.ix[i, col + '_pse'] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

    return data_table




def load_data(file_name):
    dataset = pd.read_csv(file_name)
    return dataset


def main():
    dataset = load_data("../data/processed_data.csv")
    
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0])
    
    # Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
    window_size = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(5*60000)/milliseconds_per_instance)]
    sampling_rate = float(1000)/milliseconds_per_instance
    cols = ["gFx"]
    for i in window_size:
        print("window size "+ str(i))
        data_freq = abstract_frequency(dataset, cols,i , sampling_rate)

        #print(data_freq)

        dataset.to_csv('../data/frequency_ch4_Martin.csv')


if __name__ == '__main__':
    main()
