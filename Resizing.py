import cv2 as cv
import os

def resize_and_crop_image(input_folder, target_size, destination_folder):

    """
    :param image: input image
    :param target_size: size of the cropped image, e.g. 244 for ResNet
    :return: resized and cropped image
    """
    folder = os.listdir(input_folder)
    for image in folder:
        image_path = os.path.join(input_folder, image)
        img = cv.imread(image_path)
        h, w = img.shape[:2]

        x_start = (w - h) // 2
        x_end = x_start + h
        cropped_image = img[:, x_start:x_end]

        resized_image = cv.resize(cropped_image, (target_size, target_size))

        new_image_name = os.path.splitext(image)[0] + "_resized.jpg"

        destination_path = os.path.join(destination_folder, new_image_name)

        cv.imwrite(destination_path, resized_image)


resize_and_crop_image(r"/home/martin-birch/Desktop/P3 data/Meget brugbare billeder", 224, r"/home/martin-birch/Desktop/P3 data/Resized_images")



