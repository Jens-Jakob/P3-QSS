import os
from feature_extractor import extract_features, resnet_feature_extractor, resnet152_feature_extractor
from image_loader import load_and_preprocess_image, data_loader2
from similarity_calculator import calculate_similarity
from utils import get_drone_images, get_satellite_images

# Function to extract numeric part from the filename
def extract_number_from_filename(filename):
    try:
        # Assuming the filename is something like '02321.png'
        base_name = os.path.splitext(filename)[0]
        return int(base_name)
    except ValueError:
        return None

# Function to check if the satellite image is within a certain range of the drone image number
def is_within_range(drone_filename, satellite_filename, range_value=10):
    drone_num = extract_number_from_filename(drone_filename)
    satellite_num = extract_number_from_filename(satellite_filename)

    if drone_num is not None and satellite_num is not None:
        return abs(drone_num - satellite_num) <= range_value
    return False

# Paths to data folders
drone_folder_path = r"/Users/jens-jakobskotingerslev/Desktop/P3/Dataset/vpair/queries"  # Folder containing drone images
satellite_folder_path = r"/Users/jens-jakobskotingerslev/Desktop/P3/Dataset/vpair/reference_views"  # Folder containing satellite images


def main(drone_folder_path, satellite_folder_path, top_n=3, max_images=200):
    # Initialize neural Network
    model = resnet_feature_extractor()
    # Load the selected amount of images using data_loader2
    drone_image_list = get_drone_images(drone_folder_path)
    satellite_image_list = get_satellite_images(satellite_folder_path)

    # Get the "filtered list of images"
    drone_images = data_loader2(drone_image_list, max_images)
    satellite_images = data_loader2(satellite_image_list, max_images)
    # Empty list to store the satellite features
    # Empty list to store the satellite features
    satellite_features = []

    # Extract features for all satellite images
    for sat_img in satellite_images:

        try:
            # Convert the loaded images from png to tensors
            sat_img_tensor = load_and_preprocess_image(sat_img)
            # Extract the features using the CNN
            features = extract_features(model, sat_img_tensor)
            # Append the extracted features to the list created earlier
            satellite_features.append(features)
        except Exception as e:
            print(f"Skipping unidentifiable image: {sat_img} due to error: {e}")

    # Initialize counters and results storage
    match_count = 0
    second_match_count = 0
    third_match_count = 0
    total_checked = 0
    results = []

    # Compare each drone image to the satellite images
    for drone_img in drone_images:
        total_checked += 1

        try:
            print(f"\nMatching for Drone Image: {os.path.basename(drone_img)}")
            drone_img_tensor = load_and_preprocess_image(drone_img)
            drone_features = extract_features(model, drone_img_tensor)

            # Calculate similarities with all satellite images
            similarity = calculate_similarity(drone_features, satellite_features)
            image_similar_pairs = list(zip(satellite_images, similarity))
            image_similar_pairs.sort(key=lambda x: x[1], reverse=True)

            # Print top N matches for this drone image
            print(f"Top {top_n} matches for {os.path.basename(drone_img)}:")
            for i, (sat_img, similarity) in enumerate(image_similar_pairs[:top_n]):
                sat_base_name = os.path.splitext(os.path.basename(sat_img))[0]

                # Check if satellite image is within the range of the drone image
                if is_within_range(os.path.basename(drone_img), sat_base_name):
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

# main
if __name__ == "__main__":
    main(drone_folder_path, satellite_folder_path)
