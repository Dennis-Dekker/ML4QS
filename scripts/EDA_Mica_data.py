import pandas as pd
import numpy as np
import os


def add_day_label(data, dates):
    data["day_label"] = ""
    day_labels = list(range(0,len(dates)))
    for i in range(len(data.index)):
        for date, label in zip(dates, day_labels):
            if str(data.index[i].split(" ")[0]) == date:
                data.loc[data.index[i],"day_label"] = label

    return data


def get_dates(data):
    time_points = list(data.index)
    for i in range(0, len(time_points)):
        time_points[i] = time_points[i].split(" ")[0]
    dates = np.unique(time_points)

    return dates


def load_data(path):
    data = pd.read_csv(path, index_col="time")

    return data


def main():
    path = "all_data_Mica.csv"
    data = load_data(path)

    dates = get_dates(data)

    data = add_day_label(data, dates)

    print(data.head())

    data.to_csv("all_data_Mica_labeled.csv")


if __name__ == '__main__':
    main()
