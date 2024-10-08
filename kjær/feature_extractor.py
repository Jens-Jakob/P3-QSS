# feature_extractor.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def build_feature_extractor():
    #Load ResNet50 model and use the updated 'weights' argument
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Remove the classification layer
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()  # Set the model to evaluation mode
    return model


def extract_features(model, img_tensor):
    #Extract features from the image using the model and flatten the output
    with torch.no_grad():  # Disable gradient calculation for speed
        features = model(img_tensor)
    return features.view(-1).numpy()  # Flatten to 1D and convert to numpy

