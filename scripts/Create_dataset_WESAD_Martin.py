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
            empty_dataset.loc[empty_dataset.index[i], col] = np.nanmean(relevant_rows[col])

    return empty_dataset


def load_location_files(path):
    gps_file_names = []

    for file in os.listdir(path):
        if file.endswith(".csv"):
            gps_file_names.append(file)

    print(gps_file_names)

    return gps_file_names


def read_data(path):
    BVP_data = pd.read_csv(path + "BVP.csv", skiprows=2, names=["bvp"])
    BVP_data["time"] = np.nan
    print(BVP_data.head())
    for i in range(0, len(BVP_data.index)):
        BVP_data.at[BVP_data.index[i], "time"] = i * (float(1. / 64))

    print(BVP_data.head())
    BVP_data = granulize(BVP_data, 0.1)
    print("BVP done")

    HR_data = pd.read_csv(path + "HR.csv", skiprows=2, names=["hr"])
    HR_data["time"] = np.nan
    for i in range(0, len(HR_data.index)):
        HR_data.at[HR_data.index[i], "time"] = i
    HR_data = granulize(HR_data, 0.1)
    print("HR done")

    ACC_data = pd.read_csv(path + "ACC.csv", skiprows=2, names=["acc_x", "acc_y", "acc_z"])
    ACC_data["time"] = np.nan
    for i in range(0, len(ACC_data.index)):
        ACC_data.at[ACC_data.index[i], "time"] = i * (1. / 32)
    ACC_data = granulize(ACC_data, 0.1)
    print("ACC done")

    EDA_data = pd.read_csv(path + "EDA.csv", skiprows=2, names=["eda"])
    EDA_data["time"] = np.nan
    for i in range(0, len(EDA_data.index)):
        EDA_data.at[].time[i] = i * (1. / 4)
    EDA_data = granulize(EDA_data, 0.1)
    print("EDA done")

    TEMP_data = pd.read_csv(path + "TEMP.csv", skiprows=2, names=["temp"])
    TEMP_data["time"] = np.nan
    for i in range(0, len(TEMP_data.index)):
        TEMP_data.time[i] = i * (1. / 4)
    TEMP_data = granulize(TEMP_data, 0.1)
    print("TEMP done")

    # data1 = pd.merge(BVP_data, HR_data, on="time", how="left")
    # data2 = pd.merge(data1, ACC_data, on="time", how="left")
    # data3 = pd.merge(data2, EDA_data, on="time", how="left")
    # data = pd.merge(data3, TEMP_data, on="time", how="left")

    data1 = pd.merge(BVP_data, HR_data, right_index=True)
    data2 = pd.merge(data1, ACC_data, right_index=True)
    data3 = pd.merge(data2, EDA_data, right_index=True)
    data = pd.merge(data3, TEMP_data, right_index=True)




    return data


def main():
    path = "../assignment_3/S15/Data/RAW/"

    data = read_data(path)

    print(data.head(130))

    data.to_csv("merged_data.csv", index_label="time")




if __name__ == '__main__':
    main()