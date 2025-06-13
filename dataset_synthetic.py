
import os
import glob
import random
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class ALIKEDSyntheticDataset(Dataset):
    """Alternative dataset using synthetic homography warps of single images"""
    
    def __init__(self, poses_csv: str, root_dir: str, image_size: Tuple[int, int] = (480, 640),
                 warp_strength: float = 0.1, augment: bool = True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.warp_strength = warp_strength
        self.augment = augment
        
        self.poses = self.load_poses(poses_csv)
        self.image_path_by_name = self.get_images_path_by_name()
        
        # Filter out images that don't exist
        self.valid_poses = [pose for pose in self.poses 
                           if pose['image'] in self.image_path_by_name]
        
        print(f"Loaded {len(self.valid_poses)} valid images for synthetic warping")
        
        # Data augmentation
        if self.augment:
            self.color_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
            ])
    
    def get_images_path_by_name(self):
        images_paths = glob.glob(f"{self.root_dir}/*/images/*.png")
        return {image_path.split('/')[-1]: image_path 
                for image_path in images_paths}
    
    def load_poses(self, csv_path: str) -> List[Dict]:
        """Load pose data from CSV"""
        df = pd.read_csv(csv_path)
        poses = []
        
        for _, row in df.iterrows():
            if pd.isna(row['rotation_matrix']):
                continue
                
            poses.append({
                'image': row['image_name'],
                'scene': row['scene']
            })
        
        return poses
    
    def create_random_homography(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Create a random homography for data augmentation"""
        h, w = image_shape
        
        # Corner points - ensure they are float32
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        # Add random perturbation with bounds checking
        max_perturbation = self.warp_strength * min(h, w)
        perturbation = np.random.normal(0, max_perturbation, (4, 2)).astype(np.float32)
        
        # Clamp perturbations to keep corners within reasonable bounds
        # This prevents degenerate homographies
        perturbation = np.clip(perturbation, -max_perturbation * 2, max_perturbation * 2)
        
        warped_corners = corners + perturbation
        
        # Ensure warped corners are valid (no NaN/inf) and within image bounds
        warped_corners = np.clip(warped_corners, 
                            [-w * 0.2, -h * 0.2],  # Allow some outside bounds
                            [w * 1.2, h * 1.2])
        
        # Ensure both arrays are float32 and properly shaped
        corners = corners.astype(np.float32)
        warped_corners = warped_corners.astype(np.float32)
        
        # Check for degenerate cases (points too close together)
        if np.any(np.isnan(warped_corners)) or np.any(np.isinf(warped_corners)):
            # Fallback to identity transformation
            return np.eye(3, dtype=np.float32)
        
        try:
            # Compute homography
            H = cv2.getPerspectiveTransform(corners, warped_corners)
            
            # Check if homography is valid (not singular)
            if np.any(np.isnan(H)) or np.any(np.isinf(H)):
                return np.eye(3, dtype=np.float32)
                
            return H
        except cv2.error:
            print("Homography computation failed, returning identity matrix.")
            # If homography computation fails, return identity
            return np.eye(3, dtype=np.float32)
        

    def load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)
        
        if self.augment and hasattr(self, 'color_transform'):
            image = self.color_transform(image)
        
        return image
    
    def __len__(self) -> int:
        return len(self.valid_poses)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get training sample using synthetic warping"""
        pose = self.valid_poses[idx]
        
        # Load base image
        img_path = self.image_path_by_name[pose['image']]
        image = self.load_and_preprocess_image(img_path)
        
        # Create two versions: original and warped
        H = self.create_random_homography(self.image_size)
        H_inv = np.linalg.inv(H)
        
        # Apply warp to create second image
        image_np = image.permute(1, 2, 0).numpy()
        warped_image_np = cv2.warpPerspective(image_np, H, 
                                            (self.image_size[1], self.image_size[0]))
        image1 = torch.from_numpy(warped_image_np).permute(2, 0, 1)
        
        # Create warp parameters with width and height
        warp01_params = {
            'homography': torch.from_numpy(H).float(),
            'warp_type': 'homography',
            'width': self.image_size[1],   # Width (W)
            'height': self.image_size[0]   # Height (H)
        }
        warp10_params = {
            'homography': torch.from_numpy(H_inv).float(),
            'warp_type': 'homography',
            'width': self.image_size[1],   # Width (W)
            'height': self.image_size[0]   # Height (H)
        }
        
        return {
            'image0': image,      # Original image
            'image1': image1,     # Warped image
            'warp01_params': warp01_params,
            'warp10_params': warp10_params,
            'scene': pose['scene'],
            'image_name0': pose['image'],
            'image_name1': f"{pose['image']}_warped",
        }