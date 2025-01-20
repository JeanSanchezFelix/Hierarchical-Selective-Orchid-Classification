import os
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


class MonkeypoxDataset(Dataset):


    def __init__(self,mode="train"):
        rootDir = "data/monkeypox/download/"
        self.name="MonkeyPox"
        directory=""
        if mode == "train":
            directory = "Train"
        elif mode == "test":
            directory = "Test"
        else:
            print("ERROR:Invalid mode passed")
            return
        self.dataset = datasets.ImageFolder(rootDir + directory,
                                             transform = None)
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self,idx):
        return self.dataset[idx]

    def getName(self):
        return self.name


class TestMonkeypoxDataset():
    def __init__(self,mode = "train"):
        self.numClasses = 2
        self.validationDict = {}
        self.totalImages = 0
        self.mode=mode
        if mode == "train":
            self.totalImages = 3192
            self.validationDict["Monkeypox"] = 1428
            self.validationDict["Others"] = 1764
        elif mode == "test":
            self.totalImages = 228
            self.validationDict = {}
            self.validationDict["Monkeypox"] = 102
            self.validationDict["Others"] = 126
    def runTests(self):
        ds = MonkeypoxDataset(self.mode)
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
    test = TestMonkeypoxDataset("train")
    test.runTests()
    test = TestMonkeypoxDataset("test")
    test.runTests()

if __name__=='__main__':
    testDataset()
