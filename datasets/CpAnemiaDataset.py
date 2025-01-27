from torch.utils.data import Dataset
from torchvision import datasets


class CpAnemiaDataset(Dataset):

    rootDir = "data/cp-anemia/download/"

    def __init__(self, transform):
        self.name="Cp-Anemia"
        self.dataset = datasets.ImageFolder(self.rootDir,
                                             transform = transform)
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


class TestCpAnemiaDataset():
    def __init__(self):
        self.numClasses = 2
        self.validationDict = {}
        self.totalImages = 710
        self.mode="none"
        self.validationDict["Anemic"] = 424
        self.validationDict["Non-Anemic"] = 286

    def runTests(self):
        ds = CpAnemiaDataset()
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
    test = TestCpAnemiaDataset()
    test.runTests()

if __name__=='__main__':
    testDataset()
