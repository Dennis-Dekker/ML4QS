import pandas as pd
import numpy as np
import os
from xml.dom import minidom
import datetime
import gpxpy
import matplotlib.pyplot as plt


def extract_speed(folder, gps_file):
    gps_file = open(folder + gps_file, "r")
    gps = gpxpy.parse(gps_file)
    gps = gps.tracks[0].segments[0].points
    gps_speed = []

    for point_no, point in enumerate(gps):
        gps_speed.append({"speed": point.speed_between(gps[point_no - 1])})

    return gps_speed


def parse_track(trk):
    tracks = []
    for trkseg in trk.getElementsByTagName('trkseg'):
        for trkpt in trkseg.getElementsByTagName('trkpt'):
            lat = float(trkpt.getAttribute('lat'))
            lon = float(trkpt.getAttribute('lon'))
            ele = float(trkpt.getElementsByTagName('ele')[0].firstChild.data)
            rfc3339 = trkpt.getElementsByTagName('time')[0].firstChild.data
            try:
                t = datetime.datetime.strptime(rfc3339, '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                t = datetime.datetime.strptime(rfc3339, '%Y-%m-%dT%H:%M:%SZ')
            extensions = trkpt.getElementsByTagName('extensions')[0]
            trkpt_extension = extensions.getElementsByTagName('gpxtpx:TrackPointExtension')[0]
            if trkpt_extension:
                hr = int(trkpt_extension.getElementsByTagName('gpxtpx:hr')[0].firstChild.data)

            tracks.append({'lat': lat, 'lon': lon, 'ele': ele, 'time': t, 'hr': hr})

    return tracks


def load_data(folder, gps_file):
    doc = minidom.parse(folder + gps_file)
    gpx = doc.documentElement
    for node in gpx.getElementsByTagName("trk"):
        tracks = parse_track(node)

    gps_data = pd.DataFrame(tracks, columns=["time","lat", "lon", "ele", "hr"])

    return gps_data


def load_location_files(folder):
    gps_file_names = []

    for file in os.listdir(folder):
        if file.endswith(".gpx"):
            gps_file_names.append(file)

    return gps_file_names


def main():
    # folder path
    folder = "../assignment_3/Data_Mica/"

    # list all file names
    gps_file_names = load_location_files(folder)

    # create empty dataframe
    gps_data_loc = pd.DataFrame(columns=["time","lat", "lon", "ele", "hr"])

    # get gps location from files, concatenate all files by rows (on date order)
    for gps_file in sorted(gps_file_names):
        gps_data = load_data(folder, gps_file)
        gps_data_loc = gps_data_loc.append(gps_data)

    gps_data_speed = pd.DataFrame(columns=["speed"])
    for gps_file in sorted(gps_file_names):
        gps_speed = extract_speed(folder, gps_file)
        gps_data_speed = gps_data_speed.append(gps_speed)

    gps_data_all = pd.concat([gps_data_loc, gps_data_speed], axis=1)

    plt.plot(gps_data_all["time"], gps_data_all["speed"])
    plt.show()



    gps_data_all.to_csv("all_data_Mica.csv",index=False)


if __name__ == '__main__':
    main()