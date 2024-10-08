import os
from feature_extractor import extract_features, resnet_feature_extractor, resnet152_feature_extractor
from image_loader import load_and_preprocess_image, data_loader
from similarity_calculator import calculate_similarity
from utils import get_drone_images, get_satellite_images

# Paths to data folders
drone_folder_path = r"Tysk_data\queries"  # Folder containing drone images
satellite_folder_path = r"Tysk_data\reference_views"  # Folder containing satellite images


#top_n = amount of matches printed(in image_loader.py)
def main(drone_folder_path, satellite_folder_path, top_n=3):

    #Initialize neural Network
    model = resnet152_feature_extractor()

    #Load the selected amount of images
    drone_images = data_loader(get_drone_images(drone_folder_path))
    satellite_images = data_loader(get_satellite_images(satellite_folder_path))

    #Empty list to store the satellite features
    satellite_features = []

    # Extract features for all satellite images
    for sat_img in satellite_images:

        try:
            #Convert the loaded images from png to tensors
            sat_img_tensor = load_and_preprocess_image(sat_img)
            #Extract the features using the CNN
            features = extract_features(model, sat_img_tensor)
            #Append the extracted features to the list created earlier
            satellite_features.append(features)
        except Exception as e:
            print(f"Skipping unidentifiable image: {sat_img} due to error: {e}")


    # Initialize counters and results storage
    match_count = 0
    second_match_count = 0
    third_match_count = 0
    total_checked = 0
    results = []

    #Compare each drone image to the satellite images
    #For each drone image do this
    for drone_img in drone_images:
        #Count amount of images checked
        total_checked += 1

        try:
            #Print the name of the drone image currently being image matched
            print(f"\nMatching for Drone Image: {os.path.basename(drone_img)}")
            #Convert the image from a png to a tensor
            drone_img_tensor = load_and_preprocess_image(drone_img)
            #Extract the features from the tensor
            drone_features = extract_features(model, drone_img_tensor)

            #Calculate similarities with all satellite images
            similarity = calculate_similarity(drone_features, satellite_features)
            #Make a list of the satellite image and its similarity
            image_similar_pairs = list(zip(satellite_images, similarity))
            #Sort the list by highes similarity first
            image_similar_pairs.sort(key=lambda x: x[1], reverse=True)

            # Print top N matches for this drone image
            print(f"Top {top_n} matches for {os.path.basename(drone_img)}:")
            for i, (sat_img, similarity) in enumerate(image_similar_pairs[:top_n]):
                sat_base_name = os.path.splitext(os.path.basename(sat_img))[0]

                if os.path.splitext(os.path.basename(drone_img))[0] == sat_base_name:
                    match_count += 1
                    if i == 1:
                        second_match_count += 1
                    elif i == 2:
                        third_match_count += 1

                print(f"{i + 1}: Satellite Image: {os.path.basename(sat_img)}, Similarity: {similarity:.4f}")
                results.append((os.path.basename(drone_img), os.path.basename(sat_img), similarity))

            # Print the best match (highest similarity)
            best_match_image = image_similar_pairs[0][0]
            print(
                f"\nBest match for {os.path.basename(drone_img)}: {os.path.basename(best_match_image)} with similarity: {image_similar_pairs[0][1]:.4f}")
            print("\n" + "-" * 40)  # Separator line

        except Exception as e:
            print(f"Skipping unidentifiable drone image: {drone_img} due to error: {e}")

    # Print total matches found and percentage
    if total_checked > 0:
        print(f"\nTotal perfect matches found: {match_count} out of {total_checked} checked images.")
        print(f"Percentage of perfect matches: {(match_count / total_checked) * 100:.2f}%")
        print(f"Percentage of correct match in second position: {(second_match_count / total_checked) * 100:.2f}%")
        print(f"Percentage of correct match in third position: {(third_match_count / total_checked) * 100:.2f}%")

# main, hope you know what it is by now.....
if __name__ == "__main__":
    main(drone_folder_path, satellite_folder_path)
