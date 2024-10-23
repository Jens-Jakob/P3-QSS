from Model import SiameseNetwork as Siamesenetwork
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from Data_loader import Triplet_Loader
import torch
from torch.utils.data import DataLoader
import os
from torch.nn import TripletMarginLoss
import wandb
import random

# Start a new wandb run to track this script
wandb.init(
    project="Siamesetest",  # Set the wandb project where this run will be logged
    config={  # Track hyperparameters and run metadata
        "learning_rate": 0.01,
        "architecture": "SIAMESE QUANTUM NETWORK",
        "dataset": "Tyskdata",
        "epochs": 10,
    }
)

def train(save_interval=5, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    queries_folder = '/Users/jens-jakobskotingerslev/Desktop/P3/Dataset/vpair/queries'
    reference_folder = '/Users/jens-jakobskotingerslev/Desktop/P3/Dataset/vpair/reference_views'

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Triplet_Loader(queries_folder, reference_folder, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Siamesenetwork().to(device)
    criterion = TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Average loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}] completed with Average Loss: {epoch_loss:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "learning_rate": optimizer.param_groups[0]['lr']  # Log current learning rate
        })

        # Save checkpoint if required
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"triplet_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        # Step the scheduler
        scheduler.step()

    print("Training completed.")
    # Finish the wandb run
    wandb.finish()

train()
