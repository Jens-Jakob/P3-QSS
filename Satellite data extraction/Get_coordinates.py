from geopy.distance import distance
import geopy

corner_coord = (55.172448, 14.922612)
pixel_distance_x = 250
pixel_distance_y = 250
pixel_size_cm = 12.5

pixel_size_m = pixel_size_cm / 100
distance_x_m = pixel_distance_x * pixel_size_m
distance_y_m = pixel_distance_y * pixel_size_m

start_point = geopy.point.Point(corner_coord)
middle_point_x = distance(meters=distance_x_m).destination(start_point, 90)
middle_point = distance(meters=distance_y_m).destination(middle_point_x, 180)

print(f"Middle coordinate: {middle_point.latitude}, {middle_point.longitude}")
