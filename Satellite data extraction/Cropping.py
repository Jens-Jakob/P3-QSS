import rasterio
import numpy as np

file_path = 'Satellite data extraction/Satellite data/1.tif'
folder_path = 'Satellite data extraction/Satellite data/Output folder'

img_o = rasterio.open(file_path)
img = img_o.read()
img_reduced = img[:3, :, :]
image_shape = np.shape(img_reduced)

height = image_shape[1]
width = image_shape[2]
crop_size = 480
x_start = 0
y_start = 0

profile = img_o.profile
#print(profile.crs)
profile.update(count=3)
profile.update(crs='EPSG:25832')

for i in range(20):
    new_crop_size = crop_size + crop_size
    x_start = x_start + crop_size
    y_start = y_start + crop_size

    if y_start + crop_size > height or x_start + crop_size > width:
        break
    img_cropped = img_reduced[:, y_start:y_start + crop_size, x_start:x_start + crop_size]

    output_path = f"{folder_path}/output_crop_{i}.tif"

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(img_cropped)

    print(f"Cropped image {i} saved to {output_path}")


