import os
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import cv2

class DatasetVideoTrain(data.Dataset):
    def __init__(self, opt):
        super(DatasetVideoTrain, self).__init__()
        self.opt = opt
        self.cache_data = False 
        self.n_channels = opt['n_channels'] if 'n_channels' in opt else 3
        self.num_frame = self.opt['num_frame']

        self.gt_root = opt['dataroot_gt']
        self.lq_root = opt['dataroot_lq']
        self.mask_root = opt['dataroot_mask']

        self.video_names = sorted(os.listdir(self.gt_root))
        self.paths_GT = []
        self.video_lens = {} 
        
        for vid in self.video_names:
            vid_gt_path = os.path.join(self.gt_root, vid)
            if not os.path.isdir(vid_gt_path): continue
            frames = sorted(os.listdir(vid_gt_path))
            num_frames_in_video = len(frames)
            self.video_lens[vid] = num_frames_in_video
            
            for i in range(num_frames_in_video):
                self.paths_GT.append(f"{vid}_{i}")

    def __getitem__(self, index):
        # 1. Get Paths
        key = self.paths_GT[index]
        name_a, name_b = key.rsplit('_', 1)
        center_frame_idx = int(name_b)
        num_frame = self.opt['num_frame']
        
        # Get Max Frame for this specific video
        video_name = name_a
        max_frame_idx = self.video_lens[video_name] - 1
        
        frames_idx = []
        for i in range(num_frame):
            offset = i - num_frame // 2
            idx = center_frame_idx + offset
            # Clamp index so it never goes out of bounds
            idx = max(0, min(idx, max_frame_idx))
            # Format depends on your data. 
            # If CDnet uses 6 digits (in000001.jpg), adjust here. 
            # For dummy data/standard KAIR, usually 8 digits or no leading zeros.
            # Let's assume standard 8 digits for safety, or adjust based on your filenames.
            frames_idx.append(f"{idx:08d}") 

        L_paths, H_paths, M_paths = [], [], []

        for idx in frames_idx:
            # Note: You might need to adjust extensions (.jpg vs .png) based on your real data
            H_paths.append(os.path.join(self.gt_root, video_name, f"{idx}.jpg"))
            L_paths.append(os.path.join(self.lq_root, video_name, f"{idx}.jpg"))
            M_paths.append(os.path.join(self.mask_root, video_name, f"{idx}.png"))

        # 2. Load Images
        # We check if files exist to avoid crashing on missing single frames
        img_H_list = []
        img_L_list = []
        img_M_list = []
        
        for i in range(len(H_paths)):
            if os.path.exists(H_paths[i]):
                img_H_list.append(util.imread_uint(H_paths[i], self.n_channels))
                img_L_list.append(util.imread_uint(L_paths[i], self.n_channels))
                # Check Mask exists, if not black
                if os.path.exists(M_paths[i]):
                    img_M_list.append(util.imread_uint(M_paths[i], 1))
                else:
                    # Create empty mask if missing
                    h_temp, w_temp = img_H_list[-1].shape[:2]
                    img_M_list.append(np.zeros((h_temp, w_temp), dtype=np.uint8))
            else:
                # If a frame is physically missing, reuse the previous one (temporal padding)
                img_H_list.append(img_H_list[-1] if len(img_H_list)>0 else np.zeros((256,256,3), np.uint8))
                img_L_list.append(img_L_list[-1] if len(img_L_list)>0 else np.zeros((256,256,3), np.uint8))
                img_M_list.append(img_M_list[-1] if len(img_M_list)>0 else np.zeros((256,256), np.uint8))

        # 3. Stack into Numpy Arrays
        img_H = np.stack(img_H_list, axis=0) # [Frames, H, W, 3]
        img_L = np.stack(img_L_list, axis=0)
        img_M = np.stack(img_M_list, axis=0)

        # --- THESIS ADDITION: Force Background Compression ---
        # We explicitly blur the background of the Ground Truth (img_H)
        # This teaches the model to smooth out the background, saving file size.
        
        # Ensure mask is 0 to 1
        mask_expanded = img_M.copy().astype(float) / 255.0
        # Threshold: anything above 0.5 is ROI (Keep Sharp), below is Background (Blur)
        mask_expanded[mask_expanded >= 0.5] = 1.0
        mask_expanded[mask_expanded < 0.5] = 0.0
        
        # Add channel dim for broadcasting: [Frames, H, W, 1]
        if mask_expanded.ndim == 3:
            mask_expanded = np.expand_dims(mask_expanded, axis=3)

        # Create a Blurred version of the High Quality images
        img_H_blurred = np.zeros_like(img_H)
        for f in range(img_H.shape[0]):
            # Kernel size (15,15) determines blur strength. 
            # Larger number = More blur = Smaller File Size
            img_H_blurred[f] = cv2.GaussianBlur(img_H[f], (15, 15), 0)

        # Combine: Keep Sharp ROI + Use Blurred Background
        # Formula: (Sharp * Mask) + (Blurred * (1-Mask))
        img_H = (img_H * mask_expanded) + (img_H_blurred * (1.0 - mask_expanded))
        
        # Convert back to uint8 to be safe before tensor conversion
        img_H = img_H.astype(np.uint8)
        # -------------------------------------------------------

        # 4. To Tensor
        img_H = torch.from_numpy(np.ascontiguousarray(img_H)).permute(0, 3, 1, 2).float() / 255.
        img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(0, 3, 1, 2).float() / 255.
        img_M = torch.from_numpy(np.ascontiguousarray(img_M)).permute(0, 3, 1, 2).float() / 255.

        # 5. Process Mask Weights
        weights = img_M.clone()
        weights[weights >= 0.5] = 1.0
        weights[weights < 0.5] = 0.1

        return {'L': img_L, 'H': img_H, 'M': weights, 'key': key}

    def __len__(self):
        return len(self.paths_GT)