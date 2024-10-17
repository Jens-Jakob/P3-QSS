import os
import pandas as pd
import re

def get_coordinates(file_path):
    '''
    :param file_path: file path to the csv file sent from Skywatch
    :return: csv with image number and coordinates
    '''
    df = pd.read_csv(file_path)
    columns_drop = ["height","width", "Roll", "Yaw","Pitch"]
    df.drop(columns_drop, axis=1, inplace=True)

    df.to_csv('GPS_coordinates.csv', index=False)


def get_id(image):
    '''
    :param image: input image from UAV
    :return: returns the image number in the same format as in csv
    '''
    file_name = os.path.basename(image)
    regex_number = r"(?<=0)\d{4}"
    match_number = re.search(regex_number, file_name)
    id_number = match_number[0]
    return id_number

def get_gps_from_csv(image, file):
    '''
    :param image: input image from UAV
    :param file: csv file from get_coordinates
    :return: prints the coordinates of the image
    '''
    id_number = get_id(image)
    if id_number is None:
        print("ID not found in filename")
    df = pd.read_csv(file)
    id_number = int(id_number)
    if id_number in df['i'].values:
        matching_row = df[df['i'] == id_number]
        lat = matching_row['GPS Latitude'].values[0]
        lon = matching_row['GPS Longitude'].values[0]
        print(f"GPS-coordinates for image {id_number} are {lat}, {lon}")
    else:
        print(f"ID {id_number} not found in the CSV.")


get_gps_from_csv(r"/home/martin-birch/Desktop/P3 data/Meget brugbare billeder/01910.jpg",'GPS_coordinates.csv')
