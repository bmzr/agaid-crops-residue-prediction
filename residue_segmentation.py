import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision.models.segmentation import fcn_resnet50
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Define transformations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.Resize(height=512, width=512),
    ToTensorV2(transpose_mask=True)
])

val_transform = A.Compose([
    A.Resize(height=512, width=512),
    ToTensorV2(transpose_mask=True)
])

def load_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_mask(img_path):
    mask_path = re.sub(r"(.*)(part\d+)(.*)\.jpg", r"\1res_\2\3.tif", img_path)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    mask = np.where(mask == 255, 0, 1).astype(np.uint8)
    return mask

class CropResidueDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        # Verify all images and masks exist
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            mask_path = re.sub(r"(.*)(part\d+)(.*)\.jpg", r"\1res_\2\3.tif", img_path)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
        print(f"Verified {len(image_paths)} image-mask pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = load_image(self.image_paths[idx])
            mask = load_mask(self.image_paths[idx])

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                mask = torch.tensor(mask, dtype=torch.long)

            image = image.float()
            mask = mask.long()
            return image, mask
        except Exception as e:
            print(f"Error loading image/mask pair at index {idx}: {str(e)}")
            print(f"Image path: {self.image_paths[idx]}")
            raise

def calculate_iou(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    return intersection.float() / (union.float() + 1e-6)

def save_checkpoint(epoch, model, optimizer, loss, iou, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'iou': iou,
    }, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['iou']

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                start_epoch=0, num_epochs=30):
    best_iou = 0.0
    history = {'train_iou': [], 'train_loss': [], 'val_iou': []}
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        
        progress_bar = tqdm(total=len(train_loader), 
                          desc=f'Epoch {epoch+1}/{num_epochs}',
                          unit='batch')
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            batch_start_time = time.time()
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(outputs, dim=1)
            all_train_preds.append(preds.cpu())
            all_train_targets.append(masks.cpu())
            
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'time': f'{(time.time() - batch_start_time):.1f}s'
            })
            progress_bar.update()
            
            # Save checkpoint every 20 batches
            if (batch_idx + 1) % 20 == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pth'
                save_checkpoint(epoch, model, optimizer, loss.item(), 0, checkpoint_name)
        
        progress_bar.close()
        
        # Calculate training metrics
        all_train_preds = torch.cat(all_train_preds)
        all_train_targets = torch.cat(all_train_targets)
        train_iou = calculate_iou(all_train_preds, all_train_targets)
        
        # Validation phase
        model.eval()
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                preds = torch.argmax(outputs, dim=1)
                all_val_preds.append(preds.cpu())
                all_val_targets.append(masks.cpu())
        
        all_val_preds = torch.cat(all_val_preds)
        all_val_targets = torch.cat(all_val_targets)
        val_iou = calculate_iou(all_val_preds, all_val_targets)
        
        # Update history
        history['train_iou'].append(train_iou.item())
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_iou'].append(val_iou.item())
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Train IoU: {train_iou:.4f}")
        print(f"Val IoU: {val_iou:.4f}")
        
        # Plot progress
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_iou'], label='Train IoU')
        plt.plot(history['val_iou'], label='Val IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.show()
        
        # Save if best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with Val IoU: {best_iou:.4f}")
        
        # Save epoch checkpoint
        save_checkpoint(epoch, model, optimizer, train_loss/len(train_loader), 
                       val_iou.item(), f'checkpoint_epoch_{epoch+1}.pth')

    return history

# Setup
print("Setting up data...")

data_dir = r"...label-20250201T195206Z-001\label\residue_background" # Replace with proper file path

# Find all images recursively
image_paths = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.jpg') and 'part' in file:
            full_path = os.path.join(root, file)
            # Verify corresponding mask exists
            mask_path = re.sub(r"(.*)(part\d+)(.*)\.jpg", r"\1res_\2\3.tif", full_path)
            if os.path.exists(mask_path):
                image_paths.append(full_path)

print(f"Found {len(image_paths)} valid image-mask pairs")

# Split into train and validation sets
train_images, val_images = train_test_split(image_paths, test_size=0.2, random_state=42)

# Create datasets
train_dataset = CropResidueDataset(train_images, transform=train_transform)
val_dataset = CropResidueDataset(val_images, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print("\nInitializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fcn_resnet50(weights='DEFAULT')
model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Load from specific checkpoint if needed
checkpoint_to_load = 'checkpoint_epoch_1_batch_40.pth'  # Replace with your checkpoint name

if os.path.exists(checkpoint_to_load):
    print(f"Loading checkpoint: {checkpoint_to_load}")
    start_epoch, last_loss, last_iou = load_checkpoint(model, optimizer, checkpoint_to_load)
    print(f"Resuming from epoch {start_epoch+1}, Loss: {last_loss:.4f}, IoU: {last_iou:.4f}")
else:
    print(f"No checkpoint found at {checkpoint_to_load}")
    start_epoch = 0

print("\nStarting training...")
history = train_model(model, train_loader, val_loader, criterion, optimizer, start_epoch=start_epoch)