import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DATASET_ROOT = 'dataset' 
TARGET_ROOT = 'Thesis_Data_Local'

# LIMITS
# Set to None to use ALL frames in highway, or a number (e.g., 50) for a quick test
MAX_FRAMES = 50 

# TARGET SPECIFIC VIDEO
TARGET_CATEGORY = 'baseline'
TARGET_VIDEO = 'highway'

def add_jpeg_noise(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def process_cdnet():
    # 1. Clean up old data to avoid confusion
    if os.path.exists(TARGET_ROOT):
        shutil.rmtree(TARGET_ROOT)
    
    # Create target directories
    os.makedirs(os.path.join(TARGET_ROOT, 'GT'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_ROOT, 'Mask'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_ROOT, 'LQ_ROI'), exist_ok=True)

    print(f"Scanning {SOURCE_DATASET_ROOT} for {TARGET_CATEGORY}/{TARGET_VIDEO}...")

    # Construct the exact path to highway
    video_path = os.path.join(SOURCE_DATASET_ROOT, TARGET_CATEGORY, TARGET_VIDEO)
    input_path = os.path.join(video_path, 'input')
    gt_path = os.path.join(video_path, 'groundtruth')

    if not os.path.exists(input_path):
        print(f"ERROR: Could not find folder: {input_path}")
        return

    # Create unique name for flattened structure
    unique_vid_name = f"{TARGET_CATEGORY}_{TARGET_VIDEO}"
    print(f"Processing {unique_vid_name}...")

    # Create output subfolders
    os.makedirs(os.path.join(TARGET_ROOT, 'GT', unique_vid_name), exist_ok=True)
    os.makedirs(os.path.join(TARGET_ROOT, 'Mask', unique_vid_name), exist_ok=True)
    os.makedirs(os.path.join(TARGET_ROOT, 'LQ_ROI', unique_vid_name), exist_ok=True)

    # Get frames
    frames = sorted([f for f in os.listdir(input_path) if f.endswith('.jpg') or f.endswith('.png')])
    if MAX_FRAMES:
        frames = frames[:MAX_FRAMES]

    for i, frame_file in enumerate(tqdm(frames)):
        # 1. Read Original Image (GT)
        img_gt = cv2.imread(os.path.join(input_path, frame_file))
        if img_gt is None: continue

        # 2. Read Mask
        # CDnet masks usually have 'gt' prefix: in000001.jpg -> gt000001.png
        mask_file = frame_file.replace('in', 'gt').replace('.jpg', '.png')
        mask_src = os.path.join(gt_path, mask_file)
        
        if os.path.exists(mask_src):
            mask = cv2.imread(mask_src, 0) 
        else:
            mask = np.zeros((img_gt.shape[0], img_gt.shape[1]), dtype=np.uint8)

        # 3. Create LQ ROI Image (Thesis Logic)
        h, w = img_gt.shape[:2]
        scale = 4 
        h_lq, w_lq = h // scale, w // scale
        
        # Resize GT and Mask to LQ size
        img_lq_clean = cv2.resize(img_gt, (w_lq, h_lq), interpolation=cv2.INTER_CUBIC)
        mask_lq = cv2.resize(mask, (w_lq, h_lq), interpolation=cv2.INTER_NEAREST)

        # Generate Bad Background
        img_lq_noisy = add_jpeg_noise(img_lq_clean, quality=10)

        # Mix them
        mask_norm = mask_lq.astype(float) / 255.0
        mask_norm[mask_norm < 0.2] = 0 
        mask_norm[mask_norm >= 0.2] = 1
        mask_norm = np.expand_dims(mask_norm, axis=2)

        img_lq_final = (img_lq_clean * mask_norm) + (img_lq_noisy * (1.0 - mask_norm))

        # 4. Save Everything with simple indices
        save_name_jpg = f"{i:08d}.jpg"
        save_name_png = f"{i:08d}.png"

        cv2.imwrite(os.path.join(TARGET_ROOT, 'GT', unique_vid_name, save_name_jpg), img_gt)
        cv2.imwrite(os.path.join(TARGET_ROOT, 'Mask', unique_vid_name, save_name_png), mask)
        cv2.imwrite(os.path.join(TARGET_ROOT, 'LQ_ROI', unique_vid_name, save_name_jpg), img_lq_final)

    print("Done! Data ready in 'Thesis_Data_Local'")

if __name__ == "__main__":
    process_cdnet()