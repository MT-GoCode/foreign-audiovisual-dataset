import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from data_loader import CelebADataset
from tqdm import tqdm
import os
from sklearn.metrics import precision_score, recall_score
import wandb

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
run = 22

wandb.init(project='FaceClassification', name=f'Run{run}')

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.RandomHorizontalFlip(),
    ], p=0.5),
    transforms.Resize((224, 224)),  # Resize images to 224x224 (ResNet-50 input size)
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values (ResNet-50 normalization)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (ResNet-50 input size)
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values (ResNet-50 normalization)
])

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

root_dir = '/scratch/arsh/flawless'
image_path = '/scratch/arsh/flawless/img_align_celeba/img_align_celeba'

# Create a dataset instance
train_dataset = CelebADataset(root_dir, image_path, 'train', transform=transform)
test_dataset = CelebADataset(root_dir, image_path, 'test', transform=test_transforms)

attribute_names = train_dataset.attr_df.columns
print(f"Training Size: {len(train_dataset)}, Test Size: {len(test_dataset)}")

# Define batch size for DataLoader
batch_size = 256

# Create DataLoader for training and testing sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained ResNet-50 model
resnet50_model = resnet50(pretrained=True)

num_attributes = len(attribute_names)
num_ftrs = resnet50_model.fc.in_features
resnet50_model.fc = nn.Linear(num_ftrs, num_attributes)
resnet50_model = resnet50_model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()#weight = weights)
optimizer = optim.Adam(resnet50_model.parameters(), lr=0.001)

num_epochs = 50

if not os.path.exists(f'run{run}'):
    os.mkdir(f'run{run}')

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for i, data in enumerate(pbar, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet50_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Log training loss
        running_loss += loss.item()
        wandb.log({"Training Loss": loss.item()})

        pbar.set_postfix({'Epoch': epoch + 1, 'Loss': running_loss/(i+1)})  # Update progress bar

    # Calculate precision and recall on the test set
    precision = torch.zeros(len(attribute_names), dtype=torch.float)
    recall = torch.zeros(len(attribute_names), dtype=torch.float)
    correct = torch.zeros(len(attribute_names), dtype=torch.float).to(device)
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = resnet50_model(images)
            test_loss += criterion(outputs, labels)
            predicted = torch.round(torch.sigmoid(outputs))
            # Compute precision and recall for each class
            correct += (predicted == labels).sum(dim=0)
            precision += torch.tensor(precision_score(labels.cpu(), predicted.cpu(), average=None, zero_division=0))
            recall += torch.tensor(recall_score(labels.cpu(), predicted.cpu(), average=None, zero_division=0))
            total += len(labels)

    accuracy = (correct / total) * 100
    precision /= len(test_dataloader)
    recall /= len(test_dataloader)

    # Log test loss and metrics
    wandb.log({"Test Loss": test_loss.item()})
    for idx, attribute_name in enumerate(attribute_names):
        wandb.log({f"Accuracy/{attribute_name}": accuracy[idx]})
        wandb.log({f"Precision/{attribute_name}": precision[idx]})
        wandb.log({f"Recall/{attribute_name}": recall[idx]})

    print(f'Epoch [{epoch + 1}], Total Accuracy : {accuracy.mean():.2f}%, Total Loss: {test_loss:.3f}')

    # Print precision and recall of each class
    for idx, attribute_name in enumerate(attribute_names):
        print(f'Class: {attribute_name:20s}, Accuracy: {accuracy[idx]:.2f}%, Precision: {precision[idx]*100:.2f}%, Recall: {recall[idx]*100:.2f}%')

    # Save model state
    torch.save(resnet50_model.state_dict(), f'run{run}/epoch_{epoch}_loss_{test_loss:.3f}.pth')

print('Finished Training')
