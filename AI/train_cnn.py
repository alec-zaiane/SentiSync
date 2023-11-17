import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
from CNN import EmotionCNN

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset class for FER2013
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

# Load the dataset
root_dir = 'data/train'
emotion_dataset = EmotionDataset(root_dir=root_dir, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(emotion_dataset))
val_size = len(emotion_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(emotion_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Instantiate the model and send it to the GPU
model = EmotionCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with tqdm
num_epochs = 15
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

# Save the trained model
torch.save(model.state_dict(), 'emotion_cnn_model.pth')
