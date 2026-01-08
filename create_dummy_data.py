import cv2
import numpy as np
import os

# Create dummy folders
os.makedirs('dummy_data/GT/video1', exist_ok=True)
os.makedirs('dummy_data/Mask/video1', exist_ok=True)
os.makedirs('dummy_data/LQ_ROI/video1', exist_ok=True)

# Create 6 fake frames (RVRT usually needs 6+ frames)
for i in range(6):
    # 1. Create GT (Red square)
    img_gt = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img_gt, (50+i*10, 50), (100+i*10, 100), (0, 0, 255), -1) # Moving square
    
    # 2. Create Mask (White where square is)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(mask, (50+i*10, 50), (100+i*10, 100), 255, -1)
    
    # 3. Create LQ (Blurry version)
    img_lq = cv2.resize(img_gt, (64, 64), interpolation=cv2.INTER_CUBIC)
    
    # Save files
    filename = f"{i:08d}"
    cv2.imwrite(f'dummy_data/GT/video1/{filename}.jpg', img_gt)
    cv2.imwrite(f'dummy_data/Mask/video1/{filename}.png', mask)
    cv2.imwrite(f'dummy_data/LQ_ROI/video1/{filename}.jpg', img_lq)

print("Dummy dataset created in /dummy_data")