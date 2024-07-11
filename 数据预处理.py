
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io

class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        label = np.load(self.label_paths[idx])

        # Normalize image to range [0, 1]
        image = image.astype(np.float32) / np.max(image)

        # Slice 3D image into 2D slices
        slices = []
        labels = []
        for i in range(image.shape[0]):
            slice_img = image[i, :, :]
            slice_label = label[i, :, :]

            # Apply transformations
            if self.transform:
                augmented = self.transform(image=slice_img, mask=slice_label)
                slice_img = augmented['image']
                slice_label = augmented['mask']

            slices.append(slice_img)
            labels.append(slice_label)

        return slices, labels

# Data augmentation
transform = A.Compose([
    A.Rotate(limit=90, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianNoise(var_limit=(0.01, 0.05), p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=50.0, alpha_affine=50.0, p=0.5),
    A.GridDistortion(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
    ToTensorV2()
])

# 
image_paths = ['path_to_image_1.npy', 'path_to_image_2.npy', '...']
label_paths = ['path_to_label_1.npy', 'path_to_label_2.npy', '...']

dataset = MedicalImageDataset(image_paths=image_paths, label_paths=label_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for slices, labels in dataloader:
    for i, (slice_img, slice_label) in enumerate(zip(slices, labels)):
        print(f'Slice {i} - Image shape: {slice_img.shape}, Label shape: {slice_label.shape}')
