import torch
from Model import SiameseNetwork
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
checkpoint_path = '/Users/jens-jakobskotingerslev/Documents/GitHub/P3-QSS/Tysk_siamese/checkpoints/triplet_model_epoch_10.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

query_image_path = '/Users/jens-jakobskotingerslev/Desktop/P3/Dataset/vpair/queries/00001.png'
reference_folder = '/Users/jens-jakobskotingerslev/Desktop/P3/Dataset/vpair/reference_views'

query_image = Image.open(query_image_path).convert('RGB')
query_image = transform(query_image)
query_image = query_image.unsqueeze(0).to(device)

reference_image_names = sorted([
    f for f in os.listdir(reference_folder)
    if f.endswith('.png') and not f.startswith('._')
])

reference_images = []
for img_name in reference_image_names:
    img_path = os.path.join(reference_folder, img_name)
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    reference_images.append(img)

reference_images = torch.stack(reference_images).to(device)

with torch.no_grad():
    query_embedding = model.forward_once(query_image)
    reference_embeddings = model.forward_once(reference_images)

distances = F.pairwise_distance(query_embedding.repeat(reference_embeddings.size(0), 1), reference_embeddings)

sorted_indices = torch.argsort(distances)
sorted_distances = distances[sorted_indices]
sorted_image_names = [reference_image_names[idx] for idx in sorted_indices.cpu().numpy()]

print("Ranking of reference images based on similarity to the query image:")
for rank, (img_name, distance) in enumerate(zip(sorted_image_names, sorted_distances)):
    print(f"{rank + 1}: {img_name}, Distance: {distance.item():.4f}")
