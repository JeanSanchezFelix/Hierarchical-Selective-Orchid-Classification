from torch.utils.data import Dataset
from torchvision import datasets

cancer_classes = ["Squamous cell carcinoma", "Melanoma", "Basal cell carcinoma"]
other_classes = [
    "Vascular lesions", "Monkeypox", "Actinic keratoses", "Melanocytic nevi",
    "Measles", "Healthy", "HFMD", "Cowpox", "Dermatofibroma", "Chickenpox",
    "Benign keratosis-like lesions", "Actinic keratoses"
]


class SkinCancerDataset(Dataset):

    rootDir = "data/skin-lesions/download/"

    def __init__(self, transform, mode="train", binary_mapping=False):
        self.name="Skin Lesions"
        self.dataset = datasets.ImageFolder(self.rootDir + mode, transform = transform)

        if binary_mapping:
            label_mapping = {class_name: 1 for class_name in cancer_classes}
            label_mapping.update({class_name: 0 for class_name in other_classes})
            original_classes = self.dataset.classes                                                             # Class names in the dataset
            class_to_binary_label = {original_classes.index(cls): label_mapping[cls] for cls in label_mapping}
            # Apply the label mapping dynamically to the dataset
            self.dataset.targets = [class_to_binary_label[target] for target in self.dataset.targets]
            self.dataset.classes = ['Other', 'Cancer']
            self.class_to_idx = {'Other': 0, 'Cancer': 1}
            samples = []
            # samples = self.dataset.samples
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
        return self.dataset[idx]

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
