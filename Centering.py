import cv2 as cv
import os
import numpy as np


def standardize_image(input_folder, destination_folder):
    folder = os.listdir(input_folder)
    for image in folder:
        image_path = os.path.join(input_folder, image)
        img = cv.imread(image_path).astype(np.float64)

        b, g, r = cv.split(img)

        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)

        std_b = np.std(b)
        std_g = np.std(g)
        std_r = np.std(r)

        b_standardized = (b - mean_b) / std_b
        g_standardized = (g - mean_g) / std_g
        r_standardized = (r - mean_r) / std_r

        standardized_image = cv.merge([b_standardized, g_standardized, r_standardized])
        standardized_image = np.clip(standardized_image, 0, 255).astype(np.uint8)

        new_image_name = os.path.splitext(image)[0] + "centered.jpg"

        destination_path = os.path.join(destination_folder, new_image_name)
        cv.imwrite(destination_path, standardized_image)

standardize_image(r"/home/martin-birch/Desktop/P3 data/Resized_images",
            r"/home/martin-birch/Desktop/P3 data/Resized_and_centered")

