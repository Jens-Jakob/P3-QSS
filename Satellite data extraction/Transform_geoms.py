import rasterio
from rasterio import warp, features

file_path = ('Satellite data extraction/Satellite data/Output folder/output_crop_0.tif')

with rasterio.open(file_path) as dataset:
    original_crs = dataset.crs
    mask = dataset.dataset_mask()
    transformed_geoms = []

    for geom, val in features.shapes(mask, transform=dataset.transform):
        transformed_geom = warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=6)
        transformed_geoms.append(transformed_geom)
print(geom)
print('__________________________')
print(transformed_geoms[:])
