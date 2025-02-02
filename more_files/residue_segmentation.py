import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import re
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision.models.segmentation import fcn_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# To initialize training  set DATA_DIR=C:\Users\skyli\Downloads\label-20250201T195206Z-001\label Replace with personal directory

def get_base_dir():
    # Check if an environment variable is set
    if "DATA_DIR" in os.environ:
        return os.environ["DATA_DIR"]

    # Otherwise, find the directory dynamically
    current_dir = os.getcwd()  # Get the directory where the script is running
    while current_dir:
        potential_path = os.path.join(current_dir, "label")
        if os.path.exists(potential_path):  # Check if "label" folder exists
            return potential_path
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Stop if we reach the root
            break
        current_dir = parent_dir

    raise FileNotFoundError("Could not find 'label' directory. Please set DATA_DIR environment variable.")

def get_image_paths(base_dir):
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.jpg') and 'part' in file:
                full_path = os.path.join(root, file)
                mask_path = re.sub(r"(.*)(part\d+)(.*)\.jpg", r"\1res_\2\3.tif", full_path)
                if os.path.exists(mask_path):
                    image_paths.append(full_path)
    return image_paths

class CropResidueDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_path = re.sub(r"(.*)(part\d+)(.*)\.jpg", r"\1res_\2\3.tif", self.image_paths[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 255, 0, 1).astype(np.uint8)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image.float(), mask.long()

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, checkpoint_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_preds, all_train_targets = [], []
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_train_preds.append(preds.cpu())
            all_train_targets.append(masks.cpu())
        
        train_iou = calculate_iou(torch.cat(all_train_preds), torch.cat(all_train_targets))
        
        model.eval()
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                preds = torch.argmax(outputs, dim=1)
                all_val_preds.append(preds.cpu())
                all_val_targets.append(masks.cpu())
        
        val_iou = calculate_iou(torch.cat(all_val_preds), torch.cat(all_val_targets))
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Train IoU={train_iou:.4f}, Val IoU={val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model with Val IoU: {best_iou:.4f}")
    
if __name__ == "__main__":
    base_dir = get_base_dir()
    sections = ["residue_background", "sunlit_shaded"]
    
    for section in sections:
        print(f"Training on {section}...")
        image_paths = get_image_paths(os.path.join(base_dir, section))
        train_images, val_images = train_test_split(image_paths, test_size=0.2, random_state=42)
        train_dataset = CropResidueDataset(train_images, transform=A.Compose([A.Resize(512, 512), ToTensorV2()]))
        val_dataset = CropResidueDataset(val_images, transform=A.Compose([A.Resize(512, 512), ToTensorV2()]))
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        model = fcn_resnet50(weights='DEFAULT')
        model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, checkpoint_path=f"best_model_{section}.pth")
    
    print("Training on both datasets sequentially...")
    all_image_paths = get_image_paths(base_dir)
    train_images, val_images = train_test_split(all_image_paths, test_size=0.2, random_state=42)
    train_dataset = CropResidueDataset(train_images, transform=A.Compose([A.Resize(512, 512), ToTensorV2()]))
    val_dataset = CropResidueDataset(val_images, transform=A.Compose([A.Resize(512, 512), ToTensorV2()]))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = fcn_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, checkpoint_path="best_model_combined.pth")
