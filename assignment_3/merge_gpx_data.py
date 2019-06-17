import pandas as pd
import os 
import gpxpy

def load_data(file_name, folder):
    gps_file= open(folder + file_name,"r")
    gps = gpxpy.parse(gps_file)
    gps= gps.tracks[0].segments[0].points   
    gps_data = pd.DataFrame(columns=["lon","lat","alt","time","HR"])
    
    for point in gps:
        gps_data= gps_data.append({"lon":point.longitude,"lat":point.latitude,"alt":point.elevation,"time":point.time,"hr":point.heart_rate}, ignore_index=True)
    return gps_data

def load_locations_files(folder):
    file_names= []
    for file in os.listdir(folder):
        if file.endswith(".gpx"):
            file_names.append(file)
    return file_names

def main():
    folder = "./Data_Mica/"
    gpx_files = load_locations_files(folder)     
    for i in gpx_files:
        gps_data = load_data(i,folder) 
    print(gps_data.head())
if __name__ == "__main__" :
    main()
    
