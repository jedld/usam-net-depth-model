import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torchvision.transforms import functional as F

IMAGE_SHAPE = (375, 1242)

def transform_fn():
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Resize(IMAGE_SHAPE),
            transforms.ToTensor()
    ])

def transform_seg_fn():
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Resize(IMAGE_SHAPE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

def test_transform_fn():
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SHAPE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.50625424, 0.52283798, 0.41453917], std=[0.21669488, 0.1980729 , 0.18691985])
        ])

def test_transform_seg_fn():
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SHAPE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

def convert_image(image):
    return torch.tensor(image.astype(np.float32) / 256.0).unsqueeze(0)

def transform_disparity_fn():
    return transforms.Compose([
        convert_image,
        transforms.Resize(IMAGE_SHAPE, interpolation=transforms.InterpolationMode.NEAREST)
    ])


class StereoDataset(Dataset):
    def __init__(self, left_images_folder, right_images_folder, disparity_maps_folder, sky_mask_folder=None, randomFlip=False, transform=None, transform_disparity=None):
        self.left_images_folder = left_images_folder
        self.right_images_folder = right_images_folder
        self.disparity_maps_folder = disparity_maps_folder
        self.transform = transform
        self.transform_disparity = transform_disparity
        self.randomFlip = randomFlip
        
        self.left_images = os.listdir(left_images_folder)
        self.right_images = os.listdir(right_images_folder)
        self.disparity_maps = os.listdir(disparity_maps_folder)
        
        # remove images that end with _11.png
        self.left_images = [x for x in self.left_images if not x.endswith('_11.png')]
        self.right_images = [x for x in self.right_images if not x.endswith('_11.png')]

        self.left_images.sort()
        self.right_images.sort()
        self.disparity_maps.sort()
       

        # check that the number of images is the same
        print(f'Found {len(self.left_images)} left images')
        print(f'Found {len(self.right_images)} right images')
        print(f'Found {len(self.disparity_maps)} disparity maps')
        
        assert len(self.left_images) == len(self.right_images) == len(self.disparity_maps)
        if sky_mask_folder:
            assert len(self.left_images) == len(self.sky_masks)

        
    def __len__(self):
        return len(self.left_images)
    
    def __getitem__(self, idx):
        left_image = cv2.imread(os.path.join(self.left_images_folder, self.left_images[idx]))
        right_image = cv2.imread(os.path.join(self.right_images_folder, self.right_images[idx]))
        disparity_map = cv2.imread(os.path.join(self.disparity_maps_folder, self.disparity_maps[idx]), cv2.IMREAD_UNCHANGED)
            

        left_image = self.transform(left_image)
        right_image = self.transform(right_image)
        disparity_map = self.transform_disparity(disparity_map)
        
        return left_image, right_image, disparity_map