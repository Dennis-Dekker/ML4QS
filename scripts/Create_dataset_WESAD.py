import pandas as pd
import numpy as np
import os


def load_location_files(path):
    gps_file_names = []

    for file in os.listdir(path):
        if file.endswith(".csv"):
            gps_file_names.append(file)

    return gps_file_names


def read_data(path):
    BVP_data = pd.read_csv(path + "BVP.csv", skiprows=2, names=["bvp"])
    BVP_data["time"] = np.NAN
    for i in range(0, len(BVP_data.index)):
        BVP_data.time[i] = i * (1 / 64)

    HR_data = pd.read_csv(path + "HR.csv", skiprows=2, names=["hr"])
    HR_data["time"] = np.NAN
    for i in range(0, len(HR_data.index)):
        HR_data.time[i] = i

    ACC_data = pd.read_csv(path + "ACC.csv", skiprows=2, names=["acc_x", "acc_y", "acc_z"])
    ACC_data["time"] = np.NAN
    for i in range(0, len(ACC_data.index)):
        ACC_data.time[i] = i * (1 / 32)

    EDA_data = pd.read_csv(path + "EDA.csv", skiprows=2, names=["eda"])
    EDA_data["time"] = np.NAN
    for i in range(0, len(EDA_data.index)):
        EDA_data.time[i] = i * (1 / 4)

    TEMP_data = pd.read_csv(path + "TEMP.csv", skiprows=2, names=["temp"])
    TEMP_data["time"] = np.NAN
    for i in range(0, len(TEMP_data.index)):
        TEMP_data.time[i] = i * (1 / 4)

    data = pd.merge(BVP_data,HR_data, how="left", on="time")
    data = pd.merge(data, ACC_data, how="left", on="time")
    data = pd.merge(data, EDA_data, how="left", on="time")
    data = pd.merge(data, TEMP_data, how="left", on="time")

    return data


def main():
    path = "../assignment_3/S15/S15_E4_Data/"

    data = read_data(path)

    print(data.head(130))

    data.to_csv("merged_data.csv", index_label="time")




if __name__ == '__main__':
    main()