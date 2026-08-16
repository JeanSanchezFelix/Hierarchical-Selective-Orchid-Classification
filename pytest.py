import os

from datasets.TaxonomicOrchidDataset import TaxonomicOrchidDataset  
from datasets.TaxonomicOrchidDataset import TestTaxonomicOrchidDataset
from torchvision import datasets, transforms

def main():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print("Initializing dataset...")

    dataset = TaxonomicOrchidDataset(transform, mode="train")
    testset = TestTaxonomicOrchidDataset(mode="train")

    testset.runTests()
    
    print(f"Dataset name: {dataset.name}")
    print(f"Root Directory: {dataset.getDir()}")
    
    # Print genus and species
    print("\nGenus -> Species Map:")
    for genus, species_list in dataset.getLabels().items():
        print(f"  {genus}: {species_list}")

    print(f"\nTotal classes (species): {len(dataset.class_to_idx)}")
    print(f"Total images in dataset: {len(dataset)}")

    # Preview a few items
    print("\nPreviewing first 3 samples:")
    for i in range(min(3, len(dataset))):
        img, label = dataset[i]
        print(f"Sample {i}: Image shape = {img.shape}, Label = {label}")

if __name__ == "__main__":
    main()
