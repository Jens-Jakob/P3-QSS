# feature_extractor.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def resnet_feature_extractor():
    #Load ResNet50 model and use the updated 'weights' argument

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Remove the classification layer
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()  # Set the model to evaluation mode
    return model

def retrained_feature_extractor():
    model = torch.load('resnet50_model.pth')

def extract_features(model, img_tensor):
    #Extract features from the image using the model and flatten the output
    with torch.no_grad():  # Disable gradient calculation for speed
        features = model(img_tensor)
    return features.view(-1).numpy()  # Flatten to 1D and convert to numpy




#https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html
def resnet152_feature_extractor():
    # Load ResNet152 model with pretrained weights
    model = models.resnet152(weights='IMAGENET1K_V1')

    # Remove the classification layer (the last fully connected layer)
    model.fc = nn.Identity()  # This replaces the classification layer with an identity layer

    # Create a list of layers to replace
    layers_to_replace = []

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            layers_to_replace.append((name, nn.PReLU()))

    # Replace ReLU with PReLU after the iteration
    for name, new_layer in layers_to_replace:
        setattr(model, name, new_layer)

    model.eval()  # Set the model to evaluation mode
    return model

