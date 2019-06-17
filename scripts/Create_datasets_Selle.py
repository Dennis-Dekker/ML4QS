import pandas as pd
import os
import gpxpy


def load_data(folder, measurement_file, gps_file):
    m_data = pd.read_csv(folder + measurement_file)
    gps_file = open(folder + gps_file, "r")
    gps = gpxpy.parse(gps_file)
    gps = gps.tracks[0].segments[0].points
    gps_data = pd.DataFrame(columns=["lon", "lat", "alt", "time"])

    for point in gps:
        gps_data = gps_data.append({"lon": point.longitude, "lat": point.latitude,
                                    "alt": point.elevation, "time": point.time, }, ignore_index=True)

    return m_data, gps_data


def load_location_files(folder):
    gps_file_names = []
    measurement_file_names = []

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            measurement_file_names.append(file)
        elif file.endswith(".gpx"):
            gps_file_names.append(file)

    return measurement_file_names, gps_file_names


def main():
    folder = "../assignment_3/Data_Selle/"

    measurement_file_names, gps_file_names = load_location_files(folder)

    for measurement_file, gps_file in zip(measurement_file_names, gps_file_names):
        m_data, gps_data = load_data(folder, measurement_file, gps_file)
        print(m_data.head())
        print(gps_data.head())
        print(measurement_file)
        break


if __name__ == '__main__':
    main()