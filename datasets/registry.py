from datasets import CpAnemiaDataset, MonkeypoxDataset, SkinCancerDataset, TaxonomicOrchidDataset

# Registry mapping callback names to constructors
DATASET_REGISTRY = {
    "CpAnemia": CpAnemiaDataset,
    "MonkeyPox": MonkeypoxDataset,
    "SkinCancer": SkinCancerDataset,
    "TaxonomicOrchid": TaxonomicOrchidDataset
}
