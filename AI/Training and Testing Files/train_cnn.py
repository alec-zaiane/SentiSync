import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from CNN import EmotionCNN, DeeperEmotionCNN

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset class for FER2013
class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.class_labels = self.dataset.classes

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


# # Define data augmentation and transformation
# transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.RandomHorizontalFlip(),  # Data augmentation: horizontal flip
#     transforms.RandomVerticalFlip(),  # Data augmentation: vertical flip
#     transforms.RandomRotation(degrees=15),  # Data augmentation: random rotation
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),  # Data augmentation: perspective transformation
#     transforms.Resize((48, 48)),
#     transforms.ToTensor(),
# ])


# Load the dataset
root_dir = 'C:/Users/ksekh/OneDrive/Desktop/Hackathon/SentiSync/AI/3emotions_data'
emotion_dataset = EmotionDataset(root_dir=root_dir, transform=transform)

print("Emotion Order in Training:", emotion_dataset.class_labels)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(emotion_dataset))
val_size = len(emotion_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(emotion_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Instantiate the model and send it to the GPU
# model = EmotionCNN().to(device)
model = DeeperEmotionCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)


# ReduceLROnPlateau scheduler: Reduce learning rate when a metric has stopped improving
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

# Training loop with tqdm
num_epochs = 35
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')
    for batch_idx, batch in progress_bar:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send tensors to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Update tqdm with the current loss
        progress_bar.set_postfix({'Loss': running_loss / (batch_idx + 1)})

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  # Send tensors to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print validation accuracy
    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy}%')

    # Update learning rate based on validation loss
    # val_loss = running_loss / len(val_loader)
    val_loss = running_loss
    scheduler.step(val_loss)

# Save the trained model
torch.save(model.state_dict(), 'emotion_cnn_model12.pth')