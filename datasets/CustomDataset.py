import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from collections import Counter
import random
import os

class CustomDataset(Dataset):
    """
    Base class for custom datasets with optional binary mapping and selective augmentation.

    Args:
        root_dir (str): Path to dataset.
        transform (transforms.Compose): Image transformations.
        mode (str): Dataset split ('train', 'val', 'test').
        use_minority_augmentation (bool): Whether to apply augmentations only to minority classes.
        minority_threshold (int): Threshold below which a class is considered a minority.
    """
    def __init__(self, root_dir: str, transform, mode="train", use_minority_augmentation=False, minority_threshold=1000, aug_prob=0.5):
        # self.name = "Custom Dataset"
        # self.root_dir = os.path.join(root_dir, mode)
        self.dataset = datasets.ImageFolder(os.path.join(root_dir, mode), transform=transform)
        self.use_minority_augmentation = use_minority_augmentation
        self.aug_prob = aug_prob
        
        if use_minority_augmentation:
            # Compute class distribution dynamically
            targets = np.array(self.dataset.targets)
            class_counts = Counter(targets)
            self.minority_classes = {cls for cls, count in class_counts.items() if count < minority_threshold}

            # Define augmentation only for minority classes
            self.minority_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, shear=10),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Apply augmentation with probability `self.aug_prob`
        if self.use_minority_augmentation and label in self.minority_classes and random.random() < self.aug_prob:
            img = self.minority_transform(img)

        return img, label

    def getName(self):
        return self.name
