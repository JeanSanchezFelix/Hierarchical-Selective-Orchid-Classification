import os
import torch
import numpy as np
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
    rootDir = "G:/My Drive/ColabNotebooks/TaxonomicOrchidDataset" #For testing reasons
#   root_dir = "data/taxonomic-orchid/download/"

    def __init__(self, transform, mode="train", hierarchical_class_mode=False, use_minority_augmentation=True, minority_threshold=100):
        self.name = "Taxonomic Orchid Dataset (TOD)"
        self.mode = mode

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



        # Initialize the dataset with a dummy transform
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        dataset = TaxonomicOrchidDataset(
            root_dir="data/taxonomic-orchid/download/",
            transform=transform,
            mode=mode,
            split_ratios=split_ratios,
            random_state=random_state
        )

        # Get the full dataset for reference
        full_dataset = datasets.ImageFolder(dataset.root_dir)
        targets = np.array(full_dataset.targets)

        # Compute split indices
        train_idx, temp_idx = train_test_split(
            np.arange(len(targets)),
            test_size=1 - split_ratios['train'],
            stratify=targets,
            random_state=random_state
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=split_ratios['test'] / (split_ratios['val'] + split_ratios['test']),
            stratify=targets[temp_idx],
            random_state=random_state
        )

        # Select indices based on mode
        if mode == "train":
            self.indices = train_idx
        elif mode == "val":
            self.indices = val_idx
        elif mode == "test":
            self.indices = test_idx
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.data = Subset(full_dataset, self.indices)

        # Calculate total images for this mode
        self.totalImages = len(self.indices)

        # Get species (classes) from the full dataset
        self.class_to_idx = full_dataset.class_to_idx
        self.numClasses = len(self.class_to_idx)

        # Calculate image counts per species for this mode
        self.validationDict = defaultdict(int)
        for idx in self.indices:
            _, label = full_dataset[idx]
            self.validationDict[label] += 1

    def get_images_species(self):
        """
        Returns a dictionary with image counts per species for the specified mode.
        """
        full_dataset = datasets.ImageFolder("data/taxonomic-orchid/download/")
        return {full_dataset.classes[label]: count for label, count in self.validationDict.items()}

    def get_images_genera(self):
        """
        Returns a dictionary with image counts per genus for the specified mode.
        """
        genus_counts = defaultdict(int)
        taxon = TaxonomicOrchidDataset.get_taxon_mapping("data/taxonomic-orchid/download/")
        for genus, species_list in taxon.items():
            for species in species_list:
                if species in self.class_to_idx:
                    label = self.class_to_idx[species]
                    if label in self.validationDict:
                        genus_counts[genus] += self.validationDict[label]
        return genus_counts

    def runTests(self):
        """
        Runs tests to validate the dataset: length, number of classes, and image counts per class.
        """
        ds = TaxonomicOrchidDataset(
            root_dir="data/taxonomic-orchid/download/",
            transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
            mode=self.mode,
            split_ratios=self.split_ratios,
            random_state=self.random_state
        )
        print(f"Testing {ds.name} dataset: subset {self.mode}")

        # Validate dataset length
        assert len(ds) == self.totalImages, f"Expected {self.totalImages} images, got {len(ds)}"
        print(f"\t{ds.name}:{self.mode} Length validated")

        # Validate number of classes
        assert len(ds.class_to_idx) == self.numClasses, f"Expected {self.numClasses} classes, got {len(ds.class_to_idx)}"
        print(f"\t{ds.name}:{self.mode} Number of classes validated")

        # Validate image counts per class
        dsDict = defaultdict(int)
        for i in range(len(ds)):
            _, label = ds[i]
            dsDict[label] += 1

        for label, count in self.validationDict.items():
            assert dsDict[label] == count, f"Expected {count} images for label {label}, got {dsDict[label]}"
        print(f"\t{ds.name}:{self.mode} Image qty per label validated")

def testDataset():
    split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    for mode in ["train", "val", "test"]:
        test = TestTaxonomicOrchidDataset(mode=mode, split_ratios=split_ratios)
        test.runTests()

if __name__ == '__main__':
    testDataset()