#Imports
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Define file paths
drone_folder_path = r"E:\Tysk_data\queries"  # Folder containing drone images
satellite_folder_path = r"E:\Tysk_data\reference_views"  # Folder containing satellite images


def build_feature_extractor():
    """ Load ResNet50 model and use the updated 'weights' argument """
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Remove the classification layer
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()  # Set the model to evaluation mode
    return model


def calculate_similarity(input_features, folder_features):
    """ Compare the input image's features with images in the folder """
    similarities = []
    for features in folder_features:
        similarity = cosine_similarity([input_features], [features])
        similarities.append(similarity[0][0])
    return similarities


def load_and_preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def extract_features(model, img_tensor):
    """ Extract features from the image using the model """
    with torch.no_grad():  # Disable gradient calculation for speed
        features = model(img_tensor)
    return features.squeeze().numpy()  # Remove batch dimension and convert to numpy

def Main(drone_folder_path, satellite_folder_path, top_n=10):
    """ Find the top N matching images between the drone and satellite images, ignoring non-PNG files """
    model = build_feature_extractor()

    # Get all drone images (only .png files)
    drone_images = drone_folder_path

    # Extract features for all satellite images (only .png files)
    satellite_images = [os.path.join(satellite_folder_path, img) for img in os.listdir(satellite_folder_path) if
                        img.endswith('.png') and not img.startswith('._')]
    satellite_features = []
    for sat_img_path in satellite_images:
        try:
            sat_img_tensor = load_and_preprocess_image(sat_img_path)
            features = extract_features(model, sat_img_tensor)
            satellite_features.append(features)
        except UnidentifiedImageError:
            print(f"Skipping unidentifiable image: {sat_img_path}")
            continue

    match_count = 0  # Initialize match count

    # Compare each drone image to the satellite images
    for drone_img_path in drone_images:
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

        except UnidentifiedImageError:
            print(f"Skipping unidentifiable drone image: {drone_img_path}")
            continue

    # Print total matches found
    print(f"\nTotal matches found: {match_count}")

# Call the function to find matches between drone and satellite images
Main(drone_folder_path, satellite_folder_path, top_n=10)


