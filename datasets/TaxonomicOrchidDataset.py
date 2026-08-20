import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict
from datasets.CustomHierarchicalDataset import HierarchicalDataset, OrchidTaskImageFolder
from model_compression.src.orchid.constants import TASK_FLAT_SPECIES, TASK_GENUS, TASK_GENUS_SPECIES, TASK_TARGET_GENUS

class TaxonomicOrchidDataset(HierarchicalDataset):
    """
    Custom dataset for Taxonomic Orchid multi-class classification in hierarchical or flat folder structures (e.g., Genus/Species or Species-only) with optional selective augmentation.

    Args:
        transform (`transforms.Compose`): Transformations to apply to images.
        mode (`str`): Dataset split (`train`, `val`, `test`).
        hierarchical_class_mode (`bool`):         
            - If True, expects images organized in genus/species sub-folders (2-level).
            - If False, expects species-level folders only (1-level).        
        use_minority_augmentation (`bool`): Whether to apply augmentations only to the smaller-count classes.
        minority_threshold (`int`): Classes with fewer than this number of samples are considered minority.
        allowed_classes (`list[str]`, optional): Specific species to include. Others become 'Non-Selected'.
    """
    # rootDir = "G:/My Drive/ColabNotebooks/TaxonomicOrchidDataset" #For testing reasons
    # rootDir = "data/taxonomic-orchid/download/"
    # rootDir =  "/home/jean-sanchez/Documents/TaxonomicOrchidDataset/TaxonomicOrchidDataset"
    rootDir =  str(Path("/datasets/taxomic-orchid"))

    def __init__(
        self, 
        transform=None, 
        mode="train", 
        hierarchical_class_mode=None,
        use_class_balances=True,
        use_minority_augmentation=True,
        minority_threshold=100,
        allowed_classes=None,
        root_dir=None,
        task=None,
        target_genus=None,
    ):
        self.name = "Taxonomic Orchid Dataset (TOD)"
        self.mode = mode
        self.use_minority_augmentation = use_minority_augmentation
        self.rootDir = root_dir or type(self).rootDir
        # Preserve the legacy bool option while making every new task explicit.
        if task is None:
            task = TASK_FLAT_SPECIES if hierarchical_class_mode else TASK_GENUS
        if task not in {TASK_GENUS, TASK_FLAT_SPECIES, TASK_GENUS_SPECIES, TASK_TARGET_GENUS}:
            raise ValueError(f"Unsupported orchid task '{task}'.")
        if allowed_classes is not None:
            raise ValueError("allowed_classes is not supported for explicit orchid tasks; use target_genus instead.")
        self.task = task
        self.target_genus = target_genus

        # This remains a HierarchicalDataset-compatible object: callers retain
        # ``classes``, ``targets``, class weights, and the standard __getitem__.
        self.use_class_balance = use_class_balances
        self.dataset = OrchidTaskImageFolder(
            self.rootDir,
            task=task,
            target_genus=target_genus,
            transform=transform,
        )
        self.targets = np.array(self.dataset.targets)
        self.class_to_idx = self.dataset.class_to_idx
        self.classes = self.dataset.classes
        self.aug_prob = 0.5
        self.num_classes = len(self.class_to_idx)
        self._configure_sampling(use_class_balances, use_minority_augmentation, minority_threshold)

        # Multi-class mapping
        # Get genus-to-species mapping
        self.taxon = self.taxonomic_mapping()
        
        #!NOTE: TESTING PURPOSES ONLY
        print(f"\n[TaxonomicOrchidDataset] Initialized with:")
        print(f"  - Mode: {mode}")
        print(f"  - Task: {task}{f' ({target_genus})' if target_genus else ''}")
        print(f"  - Number of classes: {len(self.classes)}")
        print(f"  - Total samples: {len(self.dataset)}")
        print(f"  - Weights: {'Class Balanced' if use_class_balances else 'Default'}")
        print(f"  - Minority Augmentation: {'Enabled' if use_minority_augmentation else 'Disabled'} (Threshold: {minority_threshold})")
        print(f"  - Classes: {self.classes}\n")

    def _configure_sampling(self, use_class_balances, use_minority_augmentation, minority_threshold):
        """Mirror the base dataset's balancing behavior for explicit task labels."""
        from collections import Counter
        from torchvision import transforms
        import random

        self._random = random
        if use_minority_augmentation:
            counts = Counter(self.targets)
            self.minority_classes = {label for label, count in counts.items() if count < minority_threshold}
            self.minority_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, shear=10),
            ])
        self.class_weights = None
        if use_class_balances:
            counts = np.bincount(self.targets, minlength=self.num_classes)
            counts = torch.tensor(counts, dtype=torch.float32)
            max_count = counts.max().clamp(min=1.0)
            beta = 1.0 - 1.0 / max_count
            effective_num = 1.0 - torch.pow(beta, counts)
            weights = (1.0 - beta) / (effective_num + 1e-8)
            self.class_weights = weights / weights.sum() * self.num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.use_minority_augmentation and label in self.minority_classes and self._random.random() < self.aug_prob:
            img = self.minority_transform(img)
        if self.use_class_balance and self.class_weights is not None:
            return img, label, self.class_weights[label]
        return img, label

    def taxonomic_mapping(self):
        """
        Walk root_path and return a dict:
            { genus_name: [ species_name, … ], … }
        Only first two levels are considered.
        Skip any folder named "UNLABELED".
        """
        taxon_map = {}
        for record in self.dataset.taxonomy.records:
            taxon_map.setdefault(record.genus_id, []).append(record.species_name)
        return taxon_map
    
    def getLabels(self):
        return self.taxon
    
    @classmethod
    def getDir(cls):
        return cls.rootDir


class TestTaxonomicOrchidDataset:
    def __init__(self, mode="train", split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1}, random_state=18):
        self.mode = mode
        self.split_ratios = split_ratios
        self.random_state = random_state

        self.ds = TaxonomicOrchidDataset(mode=self.mode)
        targets = self.ds.targets

        train_idx, temp_idx = train_test_split(
            np.arange(len(targets)),
            test_size=1 - split_ratios['train'],
            stratify=targets,
            random_state=random_state
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=split_ratios['test'] / (split_ratios['val'] + split_ratios['test']),
            # stratify=targets[temp_idx], # Commented out due to the small sample size of some species
            random_state=random_state
        )

        if mode == "train":
            self.indices = train_idx
        elif mode == "val":
            self.indices = val_idx
        elif mode == "test":
            self.indices = test_idx
        
        self.totalImages = len(self.indices)
        self.class_to_idx = self.ds.class_to_idx
        self.numClasses = len(self.class_to_idx)

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.validationDict = defaultdict(int)
        for idx in self.indices:
            label = targets[idx] 
            self.validationDict[label] += 1

        self.dataSubset = Subset(self.ds, self.indices)

    def runTests(self):
        print(("Testing %s dataset: subset %s")%(self.ds.getName(),self.mode))
        
        assert len(self.dataSubset) == self.totalImages, f"Expected {self.totalImages} images, got {len(self.dataSubset)}"
        print(("\t%s:%s Length validated")%(self.ds.getName(),self.mode))
        
        assert len(self.ds.class_to_idx) == self.numClasses, f"Expected {self.numClasses} classes, got {len(self.ds.class_to_idx)}"
        print(("\t%s:%s Num classes validated")%(self.ds.getName(),self.mode))

        dsDict = defaultdict(int)
        for i in tqdm(range(len(self.dataSubset)), desc="Processing dataset"):
            img, label = self.dataSubset[i] 
            label = int(label)
            dsDict[label] += 1
        for key, val in self.validationDict.items():
            assert dsDict[key] == val, f"Expected {val} images for label {key}, got {dsDict[key]}"

        print(("\t%s:%s Image qty per label validated")%(self.ds.getName(),self.mode))

def testDataset():
    split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    for mode in ["train", "val", "test"]:
        test = TestTaxonomicOrchidDataset(mode=mode, split_ratios=split_ratios)
        test.runTests()

if __name__ == '__main__':
    testDataset()
