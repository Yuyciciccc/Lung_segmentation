import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from AGMR_Net import EUnet  
from loss import focal_loss  
import csv
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class CTScanDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])
        
        image = Image.open(image_path).convert('RGB')  # RGB image
        label = Image.open(label_path).convert('L')  # Grayscale label
        
        # Resize the image and label to 192x192
        image = image.resize((192, 192))
        label = label.resize((192, 192))
        
        image = np.array(image)
        label = np.array(label)
        
        # Add an extra channel to the image (e.g., duplicate one of the channels)
        if image.shape[2] == 3:  # If RGB, add a fourth channel
            extra_channel = np.expand_dims(image[:, :, 0], axis=2)  # You can modify which channel to duplicate
            image = np.concatenate([image, extra_channel], axis=2)  # Now shape is (192, 192, 4)
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        # Convert to tensors and adjust dimensions
        image = torch.tensor(image, dtype=torch.float32).permute(0, 1,2) / 255.0  # Now shape is (4, 192, 192)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # Shape (1, 192, 192)

        return image, label

# Paths
image_dir = 'data2/image'
label_dir = 'data2/label'

# DataLoader with data split
transform = transforms.Compose([transforms.ToTensor()])
dataset = CTScanDataset(image_dir, label_dir, transform=transform)

# Split dataset
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# Check input size
for inputs, labels in train_loader:
    print(f'Input shape: {inputs.shape}')  # Should print torch.Size([1, 4, 192, 192])
    break

# Define the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EUnet(in_channels=4).to(device)

# Loss function
criterion = focal_loss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training parameters
num_epochs = 300
weight_path = 'log/params/AGMRnet.pth'
save_path = r'log/AGMRnet/train'
csv_path = r'log/AGMRnet/training_metrics.csv'

def print_and_save_epoch_metrics(csv_path, epoch, train_loss, val_loss):
    print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, val_loss])

def calculate_accuracy(preds, labels):
    preds = preds.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.numel()

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Ensure save path exists
os.makedirs(save_path, exist_ok=True)
all_train_losses = []  
all_val_losses = []  
all_train_accuracies = []  
all_val_accuracies = []  

# Training and validation loop
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    train_accuracies = []

    for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs, *_ = model(inputs)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        accuracy = calculate_accuracy(outputs, labels)
        train_accuracies.append(accuracy)
    
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
    
    model.eval()
    running_val_loss = 0.0
    val_accuracies = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, *_ = model(inputs)
            loss = criterion(outputs.squeeze(1), labels)
            
            running_val_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            accuracy = calculate_accuracy(outputs, labels)
            val_accuracies.append(accuracy)
    
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
    
    # Print and save metrics
    print_and_save_epoch_metrics(csv_path, epoch, epoch_train_loss, epoch_val_loss)
    
    # Save model weights
    torch.save(model.state_dict(), weight_path)
    
    # Adjust learning rate based on validation loss
    if epoch > 1 and len(all_val_losses) >= 2 and all_val_losses[-1] > all_val_losses[-2]:
        lr = optimizer.param_groups[0]['lr'] * 0.9
        adjust_learning_rate(optimizer, lr)
        print(f'Reducing learning rate to: {lr}')
    
    all_train_losses.append(epoch_train_loss)
    all_val_losses.append(epoch_val_loss)
    all_train_accuracies.append(avg_train_accuracy)
    all_val_accuracies.append(avg_val_accuracy)
    
    # Optionally save images
    # ...

# Plot training and validation losses and accuracies
plt.figure(figsize=(12, 9))
plt.plot(range(1, num_epochs + 1), all_train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), all_val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.savefig(r'log/AGMRnet/Loss.png')
plt.show()

plt.figure(figsize=(12, 9))
plt.plot(range(1, num_epochs + 1), all_train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), all_val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.savefig(r'log/AGMRnet/Accuracy.png')
plt.show()
