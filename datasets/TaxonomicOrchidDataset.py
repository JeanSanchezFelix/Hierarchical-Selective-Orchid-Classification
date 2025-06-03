import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict
from CustomHierarchicalDataset import HierarchicalDataset 

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
    """
    # rootDir = "G:/My Drive/ColabNotebooks/TaxonomicOrchidDataset" #For testing reasons
    root_dir = "data/taxonomic-orchid/download/"

    def __init__(self, transform=None, mode="train", hierarchical_class_mode=False, use_minority_augmentation=True, minority_threshold=100):
        self.name = "Taxonomic Orchid Dataset (TOD)"
        self.mode = mode
        self.use_minority_augmentation = use_minority_augmentation

        super().__init__(self.rootDir, transform, hierarchical_class_mode, use_minority_augmentation, minority_threshold)

        # Multi-class mapping
        self.class_to_idx = self.class_to_idx

        # Get genus-to-species mapping
        self.taxon = self.taxonomic_mapping()

    def taxonomic_mapping(self):
        """
        Walk root_path and return a dict:
            { genus_name: [ species_name, … ], … }
        Only first two levels are considered.
        Skip any folder named "UNLABELED".
        """
        root_path = self.rootDir 
        taxonMap = {}

        for genus_dir in sorted(os.listdir(root_path)):
            genus_path = os.path.join(root_path, genus_dir)
            
            if os.path.isdir(genus_path) and genus_dir.upper() != "UNLABELED":
                species = []

                for sp in sorted(os.listdir(genus_path)):
                    sp_path = os.path.join(genus_path, sp)
                    
                    if os.path.isdir(sp_path) and sp.upper() != "UNLABELED":
                        species.append(sp)
                
                if species:
                    taxonMap[genus_dir] = species
        return taxonMap
    
    @classmethod
    def getDir(cls):
        return cls.rootDir

    @classmethod
    def getLabels(cls):
        return cls.taxon

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