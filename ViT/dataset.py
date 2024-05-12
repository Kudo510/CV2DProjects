import torch
import os
import random
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np

class MultiRotate(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(img, angle)

class CustomDataset():
    def __init__(self, images_path, classes, data_frame):
        self.df = data_frame 
        self.data = images_path
        self.classes = classes
        self.rotation_angles = [30, 60, 90, 120, 150]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            MultiRotate(self.rotation_angles),  # Custom MultiRotate transformation
            transforms.ToTensor(),  ## Convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        label_string = self.df.loc[self.df['images'] == os.path.basename(img_path), 'labels'].values[0]
        label = torch.Tensor(np.where(self.classes == label_string)[0])
        return image, label
    