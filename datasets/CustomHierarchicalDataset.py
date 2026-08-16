import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from collections import Counter
import random
import os

Image.MAX_IMAGE_PIXELS = None

class HierarchicalImageFolder(ImageFolder):
    """
    Custom extension of torchvision.datasets.ImageFolder that supports hierarchical class structures or flat directories. 
        - NOTE: Skips folders named 'UNLABELED' at any level.
        - Base code could be expanded for multiple levels.

    Args:
        root_dir (`str`): Path to dataset.
        transform (`transforms.Compose`): Image transformations.
        hierarchical_class_mode (`bool`):         
            - If True, expects images organized in genus/species sub-folders (2-level).
            - If False, expects species-level folders only (1-level).
        allowed_classes (`list[str]`, optional): Classes to filter out. 
            - If provided, all other classes are grouped under 'Non-Selected'.
    """
    def __init__(
        self,
        root,
        transform=None,
        hierarchical_class_mode=True,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
        allowed_classes=None
    ):
        self.hierarchical_class_mode = hierarchical_class_mode
        self.allowed_classes = allowed_classes
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file
        )

    def find_classes(self, directory):
        if self.hierarchical_class_mode:
            return self.find_classes_hierarchical(directory)
        
        return self.find_classes_flat(directory)

    def find_classes_flat(self, directory):
        """
        Finds the class folders directly under `directory`, skipping 'UNLABELED'.

        Returns:
            tuple: (classes, class_to_idx)
        """
        all_classes = [
            d.name for d in os.scandir(directory)
            if d.is_dir() and d.name.upper() != "UNLABELED"
        ]
        all_classes.sort()


        if self.allowed_classes:
            # Only include specified classes, and create a 'Non-Selected' class for all others
            selected_classes = [cls for cls in all_classes if cls in self.allowed_classes]
            if not selected_classes:
                raise ValueError("No matching classes found in dataset for allowed_classes.")
            
            selected_classes.append("Non-Selected")
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(selected_classes)}
        else:
            selected_classes = all_classes
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(selected_classes)}

        return selected_classes, class_to_idx

    def find_classes_hierarchical(self, directory):
        """
        Finds all species folders under each genus folder in a hierarchy,
        skipping any 'UNLABELED' genus or species, and preserves the order of arrival.
        """
        species_order = {}
        for genus in sorted(os.listdir(directory)):
            genus_path = os.path.join(directory, genus)

            if os.path.isdir(genus_path) and genus.upper() != "UNLABELED":
                for species in sorted(os.listdir(genus_path)):
                    species_path = os.path.join(genus_path, species)
                    if (os.path.isdir(species_path) and 
                        species.upper() != "UNLABELED" and 
                        species not in species_order):
                        species_order[species] = len(species_order)
        
        species_list = list(species_order.keys())
        class_to_idx = species_order
        return species_list, class_to_idx

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None, allow_empty=False):
        """
        Builds the (image_path, class_index) dataset.
        Now properly assigns 'Non-Selected' label to classes not in allowed_classes
        """
        instances = []
        available_classes = set(class_to_idx.keys())

        if not self.hierarchical_class_mode:
            # Use genera as classes, collect images from species subfolders
            for cls in sorted(os.listdir(directory)):
                cls_path = os.path.join(directory, cls)
                if not os.path.isdir(cls_path) or cls.upper() == "UNLABELED":
                    continue

                # Determine the label name
                if self.allowed_classes:
                    # If this class is in allowed_classes, use it; otherwise use 'Non-Selected'
                    label_name = cls if cls in self.allowed_classes else "Non-Selected"
                else:
                    # No filtering - use the class name directly
                    label_name = cls
                
                # Skip if label_name is not in our class_to_idx mapping
                if label_name not in class_to_idx:
                    continue

                # Now traverse the subfolder structure (genus/species/images)
                for subfolder in sorted(os.listdir(cls_path)):
                    subfolder_path = os.path.join(cls_path, subfolder)
                    if not os.path.isdir(subfolder_path) or subfolder.upper() == "UNLABELED":
                        continue

                    for fname in sorted(os.listdir(subfolder_path)):
                        path = os.path.join(subfolder_path, fname)
                
                        # Validate file
                        if is_valid_file and not is_valid_file(path):
                            continue
                        elif extensions and not path.lower().endswith(extensions):
                            continue

                        # Add to dataset with correct label
                        item = (path, class_to_idx[label_name])
                        instances.append(item)

        else:
            # Hierarchical genus/species directory structure
            for genus in sorted(os.listdir(directory)):
                genus_path = os.path.join(directory, genus)
                if not os.path.isdir(genus_path) or genus.upper() == "UNLABELED":
                    continue

                for species in sorted(os.listdir(genus_path)):
                    species_path = os.path.join(genus_path, species)
                    if not os.path.isdir(species_path) or species.upper() == "UNLABELED":
                        continue

                    if species not in available_classes:
                        continue

                    for fname in sorted(os.listdir(species_path)):
                        path = os.path.join(species_path, fname)
                        if is_valid_file:
                            if not is_valid_file(path):
                                continue
                        elif extensions:
                            if not path.lower().endswith(extensions):
                                continue

                        item = path, class_to_idx[species]
                        instances.append(item)
                        
        self.samples = instances
        return instances


class HierarchicalDataset(Dataset):
    """
    Base class designed to handle datasets with hierarchical or flat folder structures with optional selective augmentation.
    Args:
        root_dir (`str`): Path to dataset.
        transform (`transforms.Compose`): Image transformations.
        hierarchical_class_mode (`bool`):         
            - If True, expects images organized in genus/species sub-folders (2-level).
            - If False, expects species-level folders only (1-level).        
        use_minority_augmentation (`bool`): Whether to apply augmentations only to minority classes.
        use_class_balance (`bool`): Whether to compute and use class-balanced weights.
        minority_threshold (`int`): Threshold below which a class is considered a minority.
        allowed_classes (`list[str]`, optional): Classes to include. Others become 'Non-Selected'.
    """
    def __init__(
        self,
        root_dir: str,
        transform=None,
        hierarchical_class_mode=True,
        use_minority_augmentation=False,
        use_class_balance=True,
        minority_threshold=1000,
        aug_prob=0.5,
        allowed_classes=None
    ):
        self.use_minority_augmentation = use_minority_augmentation
        self.use_class_balance = use_class_balance
        self.dataset = HierarchicalImageFolder(
            root_dir, 
            transform, 
            hierarchical_class_mode,
            allowed_classes=allowed_classes
        )
        self.targets = np.array(self.dataset.targets)
        self.class_to_idx = self.dataset.class_to_idx
        # This is taken by torchvision datasets that it also searches for the class based on the directory names
        self.classes = self.dataset.classes
        self.aug_prob = aug_prob
        self.num_classes = len(self.class_to_idx)

        if use_minority_augmentation:
            # Compute class distribution dynamically
            class_counts = Counter(self.targets)
            self.minority_classes = {cls for cls, count in class_counts.items() if count < minority_threshold}

            # Define augmentation only for minority classes
            self.minority_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, shear=10),
                # transforms.ToTensor()
            ])

        # Pre-compute class-balanced weights 
        self.class_weights = None
        if use_class_balance:
            counts = np.bincount(self.targets, minlength=self.num_classes)
            counts = torch.tensor(counts, dtype=torch.float32)

            max_count = counts.max().clamp(min=1.0)
            beta = 1.0 - 1.0 / max_count
            effective_num = 1.0 - torch.pow(beta, counts)
            class_weights = (1.0 - beta) / (effective_num + 1e-8)
            class_weights = class_weights / class_weights.sum() * self.num_classes

            self.class_weights = class_weights

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Apply augmentation with probability `self.aug_prob`
        if self.use_minority_augmentation and label in self.minority_classes and random.random() < self.aug_prob:
            img = self.minority_transform(img)

        if self.use_class_balance and self.class_weights is not None:
            balanced_weight = self.class_weights[label]
            return img, label, balanced_weight

        return img, label

    def getName(self):
        return self.name
