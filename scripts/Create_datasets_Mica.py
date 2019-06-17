import pandas as pd
import os
from xml.dom import minidom
import datetime

def parseTrack(trk):
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
            trkPtExtension = extensions.getElementsByTagName('gpxtpx:TrackPointExtension')[0]
            if trkPtExtension:
                hr = int(trkPtExtension.getElementsByTagName('gpxtpx:hr')[0].firstChild.data)

            tracks.append({'lat': lat, 'lon': lon, 'ele': ele, 'time': t, 'hr': hr})

    return tracks

def load_data(folder, gps_file):
    doc = minidom.parse(folder + gps_file)
    gpx = doc.documentElement
    for node in gpx.getElementsByTagName("trk"):
        tracks = parseTrack(node)

    gps_data = pd.DataFrame(tracks, columns=["time","lat", "lon", "ele", "hr"])

    return gps_data


def load_location_files(folder):
    gps_file_names = []

    for file in os.listdir(folder):
        if file.endswith(".gpx"):
            gps_file_names.append(file)

    return gps_file_names


def main():
    folder = "../assignment_3/Data_Mica/"

    gps_file_names = load_location_files(folder)

    gps_data_all = pd.DataFrame(columns=["time","lat", "lon", "ele", "hr"])

    for gps_file in sorted(gps_file_names):
        gps_data = load_data(folder, gps_file)
        print(gps_data.shape)
        gps_data_all = gps_data_all.append(gps_data)
    print(gps_data_all.shape)

    gps_data_all.to_csv("all_data_Mica.csv",index=False)


if __name__ == '__main__':
    main()