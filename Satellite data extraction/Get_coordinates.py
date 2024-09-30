from geopy.distance import distance
import geopy
import rasterio
from rasterio import warp, features
import pandas as pd

file_path = ('/Users/jens-jakobskotingerslev/Desktop/P3/Kodning/Billede_data/Tester1.tif')

with rasterio.open(file_path) as dataset:
    original_crs = dataset.crs
    mask = dataset.dataset_mask()
    transformed_geoms = []

    for geom, val in features.shapes(mask, transform=dataset.transform):
        transformed_geom = warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=6)
        transformed_geoms.append(transformed_geom)

corner_coord = (transformed_geoms[0]['coordinates'][0][0])
pixel_distance = 250
pixel_size_cm = 12.5

pixel_size_m = pixel_size_cm / 100
distance_x_m = pixel_distance * pixel_size_m
distance_y_m = pixel_distance * pixel_size_m

start_point = geopy.point.Point(corner_coord)
num_x = 4
num_y = 4
coordinates = []
fields = []

for i in range(num_y):
    for j in range(num_x):
        middle_point_x = distance(meters=j * distance_x_m).destination(start_point, 90)
        middle_point = distance(meters=i * distance_y_m).destination(middle_point_x, 180)
        coordinates.append((middle_point.latitude, middle_point.longitude))
        fields.append(f"Square ({i+1}, {j+1}) center coordinate:")
        print(f"Square ({i+1}, {j+1}) center coordinate: {middle_point.latitude}, {middle_point.longitude}")

#for i in range(coordinates):

