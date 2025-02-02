import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import os
from glob import glob
from tqdm import tqdm

def load_model_from_checkpoint(checkpoint_path):
    model = fcn_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_mask_path(image_path):
    """Generate the corresponding mask save path for an image"""
    dir_path = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    
    # Split the filename at 'part'
    before_part = base_name.split('part')[0]
    part_num = base_name.split('part')[1]
    
    # Insert 'res_' before 'part'
    mask_name = f"{before_part}res_part{part_num}"
    mask_path = os.path.join(dir_path, mask_name)
    
    return mask_path

def process_images(model, image_paths, save_dir='results'):
    """Process images and generate masks"""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Process each image
    for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)['out']
            prediction = torch.argmax(output, dim=1).squeeze().numpy()
        
        # Convert prediction to mask image (0-255 range)
        mask = (prediction > 0.5).astype(np.uint8) * 255
        
        # Save the mask
        mask_path = generate_mask_path(image_path)
        cv2.imwrite(mask_path, mask)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax2.imshow(mask, cmap='gray')
        ax2.set_title('Generated Mask')
        
        # Save visualization
        plt.savefig(os.path.join(save_dir, f'vis_{idx:03d}.png'))
        plt.close()
        
        print(f"Processed {os.path.basename(image_path)} → {os.path.basename(mask_path)}")

def process_sections(base_dir, model, sections):
    """Process multiple sections of data"""
    for section in sections:
        print(f"\nProcessing section: {section}")
        section_dir = os.path.join(base_dir, section)
        save_dir = os.path.join('results', section)
        
        if not os.path.exists(section_dir):
            print(f"Section directory not found: {section_dir}")
            continue
            
        # Look specifically for IMG_*_part*.jpg pattern
        image_paths = sorted(glob(os.path.join(section_dir, "IMG_*_part*.jpg")))
        if not image_paths:
            print(f"No images found in section: {section}")
            continue
            
        process_images(model, image_paths, save_dir)

def main():
    # Configuration
    checkpoint_path = 'checkpoint_epoch_2.pth' # Replace this with the most trained model checkpoint
    base_dir = r"C:\Users\skyli\Downloads\test-20250202T021914Z-001\test" # Replace with the proper Dir path
    
    # List of sections to process
    sections = [
        'Zak-W-winterBarley_1m_20220401',
        'Ritzville2-SprWheat1m20220329',
        'Ritzville3-WheatFallow1pass1m20220329',
        'Limbaugh1-1m20220328',
        'Ritzville6-SprWheatWintPeas1m20220329'
    ]
    
    # Load model
    print("Loading model...")
    model = load_model_from_checkpoint(checkpoint_path)
    
    # Process all sections
    print("Starting mask generation...")
    process_sections(base_dir, model, sections)
    print("\nProcessing complete!")

if __name__ == '__main__':
    main()
