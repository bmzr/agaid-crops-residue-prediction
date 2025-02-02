import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import torchvision.transforms as transforms
import re
import torch.nn as nn

def load_model_from_checkpoint(checkpoint_path):
    # Initialize model
    model = fcn_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def visualize_prediction(image_path, model):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))  # Add resize to match training
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    
    # Get mask path and load ground truth
    mask_path = re.sub(r"(.*)(part\d+)(.*)\.jpg", r"\1res_\2\3.tif", image_path)
    ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.resize(ground_truth, (512, 512))  # Add resize to match
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)['out']
        prediction = torch.argmax(output, dim=1).squeeze().numpy()
    
    # Calculate IoU for this prediction
    pred_binary = prediction > 0.5
    gt_binary = ground_truth > 127
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / union if union > 0 else 0
    
    # Display results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(ground_truth, cmap='gray')
    ax2.set_title('Ground Truth')
    ax3.imshow(prediction, cmap='gray')
    ax3.set_title(f'Prediction (IoU: {iou:.3f})')
    plt.show()

# Example usage
# Use the full path to your checkpoint file
checkpoint_path = r"C:\Users\skyli\Downloads\checkpoint_epoch_2.pth"  # Adjust this path to where your checkpoint is saved
model = load_model_from_checkpoint(checkpoint_path)

# Choose an image to visualize (using raw string)
image_path = r"C:\Users\skyli\Downloads\label-20250201T195206Z-001\label\residue_background\Ritzville2-SprWheat1m20220329\IMG_0784\IMG_0784_part02.jpg"
visualize_prediction(image_path, model)