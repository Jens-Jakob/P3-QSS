# main.py
import os
from feature_extractor import build_feature_extractor, extract_features
from image_loader import load_and_preprocess_image
from similarity_calculator import calculate_similarity
from utils import get_drone_images, get_satellite_images
import random

def get_subset_of_images(image_list, fraction=0.1):
    """ Get a subset of images from the original list """
    return random.sample(image_list, int(len(image_list) * fraction))

def Main(drone_folder_path, satellite_folder_path, top_n=10):
    """ Find the top N matching images between the drone and satellite images """
    model = build_feature_extractor()

    # Get all drone images (only .png files)
    drone_images = get_drone_images(drone_folder_path)
    drone_images = get_subset_of_images(drone_images, fraction=0.1)

    # Extract features for all satellite images (only .png files)
    satellite_images = get_satellite_images(satellite_folder_path)
    #satellite_images = get_subset_of_images(satellite_images, fraction=0.1)
    satellite_features = []

    for sat_img_path in satellite_images:
        try:
            sat_img_tensor = load_and_preprocess_image(sat_img_path)
            features = extract_features(model, sat_img_tensor)
            satellite_features.append(features)
        except Exception as e:
            print(f"Skipping unidentifiable image: {sat_img_path} due to error: {e}")
            continue

    match_count = 0  # Initialize match count
    total_checked = 0  # Initialize total checked count

    # Compare each drone image to the satellite images
    for drone_img_path in drone_images:
        total_checked += 1  # Increment the total checked count
        try:
            print(f"\nMatching for Drone Image: {drone_img_path}")
            drone_img_tensor = load_and_preprocess_image(drone_img_path)
            drone_features = extract_features(model, drone_img_tensor)

            # Calculate similarities with all satellite images
            similarities = calculate_similarity(drone_features, satellite_features)

            # Combine satellite images and their similarities into a list of tuples
            image_similarity_pairs = list(zip(satellite_images, similarities))

            # Sort by similarity in descending order
            image_similarity_pairs.sort(key=lambda x: x[1], reverse=True)

            # Get the base name of the drone image without extension
            drone_base_name = os.path.splitext(os.path.basename(drone_img_path))[0]

            # Print the top N satellite matches for this drone image
            print(f"Top {top_n} matches for {os.path.basename(drone_img_path)}:")
            for i, (sat_img_path, similarity) in enumerate(image_similarity_pairs[:top_n]):
                sat_base_name = os.path.splitext(os.path.basename(sat_img_path))[0]  # Base name of satellite image

                # Check for match
                if drone_base_name == sat_base_name:
                    match_count += 1  # Increment match count

                print(f"Rank {i + 1}: Satellite Image: {sat_img_path}, Similarity: {similarity}")

            # Print the best match (highest similarity)
            best_match_image = image_similarity_pairs[0][0]
            print(
                f"\nBest match for {os.path.basename(drone_img_path)}: {best_match_image} with similarity: {image_similarity_pairs[0][1]}")

        except Exception as e:
            print(f"Skipping unidentifiable drone image: {drone_img_path} due to error: {e}")
            continue

    # Print total matches found and percentage
    if total_checked > 0:
        percentage_matches = (match_count / total_checked) * 100
        print(f"\nTotal matches found: {match_count} out of {total_checked} checked images.")
        print(f"Percentage of matches: {percentage_matches:.2f}%")
    else:
        print("\nNo drone images were checked.")


# Call the function to find matches between drone and satellite images
if __name__ == "__main__":
    drone_folder_path = r"Tysk_data\queries"  # Folder containing drone images
    satellite_folder_path = r"Tysk_data\reference_views"  # Folder containing satellite images
    Main(drone_folder_path, satellite_folder_path, top_n=10)
