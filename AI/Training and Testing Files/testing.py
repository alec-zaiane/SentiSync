import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from CNN import EmotionCNN  # Assuming EmotionCNN is your custom CNN model
import matplotlib.pyplot as plt
import numpy as np


class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# Define a simple transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

root_dir = 'data/test'
dataset = EmotionDataset(root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Instantiate your model
model = EmotionCNN()

# Load the trained weights
model.load_state_dict(torch.load('emotion_cnn_model2.pth'))
model.eval()  # Set the model to evaluation mode

# Define a device for inference (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Testing loop with tqdm loading bar
correct_predictions = 0
total_samples = 0

with torch.no_grad(), tqdm(total=len(dataloader), desc='Testing') as pbar:
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Get predictions
        _, predicted = torch.max(outputs, 1)

        # Update counts
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Visualize images and predicted labels
        for i in range(inputs.size(0)):
            image = np.transpose(inputs[i].cpu().numpy(), (1, 2, 0))  # Convert to (H, W, C) for visualization
            plt.imshow(image.squeeze(), cmap='gray')
            true_label = dataset.dataset.classes[labels[i].item()]
            predicted_label = dataset.dataset.classes[predicted[i].item()]
            plt.title(f'True: {true_label}, Predicted: {predicted_label}')
            plt.show()

        # Update loading bar
        pbar.update(1)

# Calculate accuracy
accuracy = correct_predictions / total_samples
print(f'Test Accuracy: {accuracy * 100:.2f}%')
