import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

#Check if GPU is available (and which).
print(torch.cuda.is_available())  #Prints whether CUDA is available.
print(torch.cuda.device_count())  # Prints the number of GPU.
print(torch.cuda.get_device_name(0))  # Prints the name of the GPU.

#Define which image transformations will be applied to the data.
transform = transforms.Compose([
    #resize images to 224x224 pixels.
    transforms.Resize((224, 224)),
    #Converting images to PyTorch tensors.
    transforms.ToTensor(),
    #Normalize images.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])


#Function to load images
def load_images(satellite_dir, drone_dir, non_matching_ratio=3):
    #Load satellite images that is a png and does not start with (._)
    satellite_images = [f for f in os.listdir(satellite_dir) if
                        f.endswith('.png') and not f.startswith('._')]

    #Load drone images that is a png and does not start with (._)
    drone_images = [f for f in os.listdir(drone_dir) if
                    f.endswith('.png') and not f.startswith('._')]

    #Calculate total length considering non-matching pairs based on the specified ratio
    total_length = int(len(satellite_images) * (1 + non_matching_ratio))
    print(f"Dataset length = {total_length}")

    #Return lists of image filenames and total length
    return satellite_images, drone_images, total_length


#Function to create pairs of satellite and drone images with labels
def get_image_pair(index, satellite_images, drone_images):
    #Determine if we are creating a matching or non-matching pair based on index
    if index < len(satellite_images):
        img_name = satellite_images[index]  # Get corresponding satellite image name
        label = 1  # Label as matching pair (1)
    else:
        img_name = random.choice(satellite_images)  # Randomly select a satellite image for non-matching pair
        label = 0  # Label as non-matching pair (0)

    # Construct full path for the satellite image and load it
    sat_img_path = os.path.join(satellite_dir, img_name)
    sat_image = Image.open(sat_img_path)

    # Determine corresponding drone image path based on label
    if label == 1:
        drone_img_path = os.path.join(drone_dir, img_name)  # Matching drone image path
    else:
        random_drone_img_name = random.choice(drone_images)  # Randomly select a drone image for non-matching pair
        drone_img_path = os.path.join(drone_dir, random_drone_img_name)

    # Load the corresponding drone image
    drone_image = Image.open(drone_img_path)

    # Apply transformations to both images and return them along with the label as a tensor
    return transform(sat_image), transform(drone_image), torch.tensor(label, dtype=torch.float32)


# Specify directories containing satellite and drone images
satellite_dir = r'Tysk_data\reference_views'
drone_dir = r'Tysk_data\queries'

# Load images from directories and get their filenames along with total dataset length
satellite_images, drone_images, dataset_length = load_images(satellite_dir, drone_dir)

# Create dataset by generating pairs of images with labels using a list comprehension
dataset = [(get_image_pair(i, satellite_images, drone_images)) for i in range(dataset_length)]

# Calculate sizes for training, validation, and test datasets (80%, 10%, 10%)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset into training, validation, and test sets using random_split method
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders for each dataset split to facilitate batch processing during training/evaluation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle training data for better generalization
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No shuffling needed for validation data
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No shuffling needed for test data


# Function to create a ResNet model without the final classification layer (feature extractor)
def create_model():
    resnet = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet50 model weights from ImageNet
    resnet.fc = nn.Identity()  # Remove the final fully connected layer to use ResNet as a feature extractor
    return resnet


# Instantiate the model and fully connected layer for classification based on extracted features
model = create_model()
fc_layer = nn.Linear(2 * 2048, 1)  # Fully connected layer to combine features from both images

# Move model and fully connected layer to GPU if available; otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
fc_layer.to(device)

# Define loss function and optimizer for training process
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss for binary classification task
optimizer = torch.optim.Adam(list(model.parameters()) + list(fc_layer.parameters()), lr=0.001)  # Adam optimizer

# Training loop over specified number of epochs
num_epochs = 11
for epoch in range(num_epochs):
    model.train()  # Set model to training mode (enables dropout layers etc.)
    train_loss = 0  # Initialize training loss accumulator

    for sat_img, drone_img, label in train_loader:
        sat_img, drone_img, label = sat_img.to(device), drone_img.to(device), label.to(
            device)  # Move data to GPU if available

        optimizer.zero_grad()  # Clear gradients from previous step

        # Forward pass: extract features from both images using the model
        sat_features = model(sat_img).view(sat_img.size(0), -1)
        drone_features = model(drone_img).view(drone_img.size(0), -1)

        combined_features = torch.cat((sat_features, drone_features), dim=1)  # Combine features from both images

        outputs = fc_layer(combined_features)  # Pass combined features through fully connected layer

        # Calculate loss based on predictions and true labels
        loss = criterion(outputs.squeeze(), label)
        train_loss += loss.item()  # Accumulate training loss

        loss.backward()  # Backpropagation: compute gradients
        optimizer.step()  # Update model parameters using computed gradients

    # Validation step after each epoch
    model.eval()  # Set model to evaluation mode (disables dropout layers etc.)
    val_loss = 0  # Initialize validation loss accumulator

    with torch.no_grad():  # Disable gradient calculation during validation
        for sat_img, drone_img, label in val_loader:
            sat_img, drone_img, label = sat_img.to(device), drone_img.to(device), label.to(device)

            sat_features = model(sat_img).view(sat_img.size(0), -1)
            drone_features = model(drone_img).view(drone_img.size(0), -1)
            combined_features = torch.cat((sat_features, drone_features), dim=1)
            outputs = fc_layer(combined_features)

            loss = criterion(outputs.squeeze(), label)
            val_loss += loss.item()  # Accumulate validation loss

    # Print average losses after each epoch
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')



# Save the model's state dictionary after training is complete
model_save_path = 'resnet50_model.pth'
torch.save({
               'model_state_dict': model.state_dict(),  # Save state dict of ResNet model
               'fc_layer_state_dict': fc_layer.state_dict(),  # Save state dict of fully connected layer
               'optimizer_state_dict': optimizer.state_dict(), # Save state dict of optimizer
}, model_save_path)

print(f'Model saved to {model_save_path}')  # Confirmation message after saving the model
