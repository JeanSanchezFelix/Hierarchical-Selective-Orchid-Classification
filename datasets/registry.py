from datasets import CpAnemiaDataset, MonkeypoxDataset, SkinCancerDataset

# Registry mapping callback names to constructors
DATASET_REGISTRY = {
    "CpAnemia": CpAnemiaDataset,
    "MonkeyPox": MonkeypoxDataset,
    "SkinCancer": SkinCancerDataset,
}
