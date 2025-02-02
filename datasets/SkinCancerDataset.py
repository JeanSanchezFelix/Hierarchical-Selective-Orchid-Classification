import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from collections import Counter

cancer_classes = ["Squamous cell carcinoma", "Melanoma", "Basal cell carcinoma"]
other_classes = [
    "Vascular lesions", "Monkeypox", "Actinic keratoses", "Melanocytic nevi",
    "Measles", "Healthy", "HFMD", "Cowpox", "Dermatofibroma", "Chickenpox",
    "Benign keratosis-like lesions", "Actinic keratoses"
]


class SkinCancerDataset(Dataset):
    """
    Custom dataset for skin cancer classification with optional binary mapping and selective augmentation.

    Args:
        transform (transforms.Compose): Transformations to apply to images.
        mode (str): Dataset split ('train', 'val', 'test').
        binary_mapping (bool): Whether to map labels to binary classification.
        use_minority_augmentation (bool): Whether to apply augmentations only to minority classes.
        minority_threshold (int): Classes with fewer than this number of samples are considered minority.
    """
    rootDir = "data/skin-lesions/download/"

    def __init__(self, transform, mode="train", binary_mapping=True, use_minority_augmentation=False, minority_threshold=1000):
        self.name="Skin Lesions"
        self.dataset = datasets.ImageFolder(self.rootDir + mode, transform = transform)
        self.use_minority_augmentation = use_minority_augmentation
        
        # Compute class distribution dynamically
        self.targets = np.array(self.dataset.targets)
        class_counts = Counter(self.targets)
        self.minority_classes = {cls for cls, count in class_counts.items() if count < minority_threshold}

        # Define augmentation only for minority classes
        self.minority_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.ToTensor()
        ])

        if binary_mapping:
            label_mapping = {class_name: 1 for class_name in cancer_classes}
            label_mapping.update({class_name: 0 for class_name in other_classes})
            original_classes = self.dataset.classes                                                             # Class names in the dataset
            class_to_binary_label = {original_classes.index(cls): label_mapping[cls] for cls in label_mapping}

            # Apply the label mapping dynamically to the dataset
            self.dataset.targets = [class_to_binary_label[target] for target in self.dataset.targets]
            self.dataset.classes = ['Other', 'Cancer']
            self.class_to_idx = {'Other': 0, 'Cancer': 1}

            # Adjust dataset samples
            samples = []
            for sample in self.dataset.samples:
                if any(sample[0].__contains__(cancer_type) for cancer_type in cancer_classes):
                    samples.append((sample[0], 1))
                else:
                    samples.append((sample[0], 0))
            self.dataset.samples = samples
        else:
            self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        img, label = self.dataset[idx]

        # Apply augmentation only if minority augmentation is enabled and class is in minority
        if self.use_minority_augmentation and label in self.minority_classes:
            img = self.minority_transform(img)

        return img, label

    def getName(self):
        return self.name
    
    @classmethod
    def getDir(cls):
        return cls.rootDir

class TestSkinCancerDataset():
    def __init__(self,mode="train"):
        self.totalImages = 0
        self.numClasses = 14
        self.validationDict = {}
        self.mode = mode
        if mode == "train":
            self.totalImages = 29322
            self.validationDict["Actinic keratoses"] = 693
            self.validationDict["Basal cell carcinoma"] = 2658
            self.validationDict["Benign keratosis-like lesions"] = 2099
            self.validationDict["Chickenpox"] = 900
            self.validationDict["Cowpox"] = 792
            self.validationDict["Dermatofibroma"] = 191
            self.validationDict["Healthy"] = 1368
            self.validationDict["HFMD"] = 1932
            self.validationDict["Measles"] = 660
            self.validationDict["Melanocytic nevi"] = 10300
            self.validationDict["Melanoma"] = 3617
            self.validationDict["Monkeypox"] = 3408
            self.validationDict["Squamous cell carcinoma"] = 502
            self.validationDict["Vascular lesions"] = 202
        elif mode == "test":
            self.totalImages = 3674
            self.validationDict = {}
            self.validationDict["Actinic keratoses"] = 88
            self.validationDict["Basal cell carcinoma"] = 333
            self.validationDict["Benign keratosis-like lesions"] = 263
            self.validationDict["Chickenpox"] = 113
            self.validationDict["Cowpox"] = 99
            self.validationDict["Dermatofibroma"] = 25
            self.validationDict["Healthy"] = 171
            self.validationDict["HFMD"] = 242
            self.validationDict["Measles"] = 83
            self.validationDict["Melanocytic nevi"] = 1288
            self.validationDict["Melanoma"] = 453
            self.validationDict["Monkeypox"] = 426
            self.validationDict["Squamous cell carcinoma"] = 64
            self.validationDict["Vascular lesions"] = 26
        elif mode == "val":
            self.totalImages = 3660
            self.validationDict = {}
            self.validationDict["Actinic keratoses"] = 86
            self.validationDict["Basal cell carcinoma"] = 332
            self.validationDict["Benign keratosis-like lesions"] = 262
            self.validationDict["Chickenpox"] = 112
            self.validationDict["Cowpox"] = 99
            self.validationDict["Dermatofibroma"] = 23
            self.validationDict["Healthy"] = 171
            self.validationDict["HFMD"] = 241
            self.validationDict["Measles"] = 82
            self.validationDict["Melanocytic nevi"] = 1287
            self.validationDict["Melanoma"] = 452
            self.validationDict["Monkeypox"] = 426
            self.validationDict["Squamous cell carcinoma"] = 62
            self.validationDict["Vascular lesions"] = 25

    def runTests(self):
        ds = SkinCancerDataset(self.mode)
        print(("Testing %s dataset: subset %s")%(ds.getName(),self.mode))
        assert len(ds) == self.totalImages
        print(("\t%s:%s Length validated")%(ds.getName(),self.mode))
        dsDict = {}
        assert len(ds.class_to_idx) == self.numClasses
        print(("\t%s:%s Num classes validated")%(ds.getName(),self.mode))
        for i in range(len(ds)):
            img,label = ds[i]
            if label in dsDict.keys():
                dsDict[label] = dsDict[label] + 1
            else:
                dsDict[label] = 1
        for key,val in self.validationDict.items():
            assert dsDict[ds.class_to_idx[key]] == val
        print(("\t%s:%s Image qty per label validated")%(ds.getName(),self.mode))

def testDataset():
    test = TestSkinCancerDataset("train")
    test.runTests()
    test = TestSkinCancerDataset("test")
    test.runTests()
    test = TestSkinCancerDataset("val")
    test.runTests()

if __name__=='__main__':
    testDataset()
