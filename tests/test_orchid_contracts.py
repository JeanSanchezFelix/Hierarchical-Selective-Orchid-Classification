import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from datasets.TaxonomicOrchidDataset import TaxonomicOrchidDataset
from model_compression.src.orchid.checkpoints import OrchidModelCheckpoint, load_orchid_checkpoint
from model_compression.src.utils.preprocessing import load_data


class TestOrchidContracts(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        labels = {"Alpha": ["one", "two"], "Beta": ["three"]}
        rows = []
        for genus, species_list in labels.items():
            for species in species_list:
                for number, split in enumerate(("train", "val", "test")):
                    relative = Path(genus) / species / f"{number}.png"
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8), color=(number * 10, 0, 0)).save(target)
                    rows.append({"image_path": relative.as_posix(), "genus_id": genus, "species_name": species,
                                 "species_id": f"{genus}::{species}", "split": split, "split_note": ""})
        manifest = root / "split.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_task_modes_and_manifest_filtered_expert(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "orchids"
            manifest = self.make_dataset(root)
            router = TaxonomicOrchidDataset(root_dir=str(root), task="genus", use_class_balances=False, use_minority_augmentation=False)
            flat = TaxonomicOrchidDataset(root_dir=str(root), task="flat_species", use_class_balances=False, use_minority_augmentation=False)
            expert = TaxonomicOrchidDataset(root_dir=str(root), task="genus_species", target_genus="Alpha", use_class_balances=False, use_minority_augmentation=False)
            self.assertEqual(router.classes, ["Alpha", "Beta"])
            self.assertEqual(len(flat.classes), 3)
            self.assertEqual(expert.classes, ["Alpha::one", "Alpha::two"])
            loaders = load_data("TaxonomicOrchid", batch_size=2, split_manifest=str(manifest), dataset_kwargs={
                "root_dir": str(root), "task": "genus_species", "target_genus": "Alpha", "use_class_balances": False,
                "use_minority_augmentation": False,
            })
            self.assertEqual({name: len(loader.dataset) for name, loader in loaders.items()}, {"train": 2, "val": 2, "test": 2})

    def test_checkpoint_requires_deployment_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            metadata = {"task": "genus", "class_labels": ["Alpha", "Beta"], "model_name": "mobilenet_v2", "img_size": 224,
                        "normalization": {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}}
            callback = OrchidModelCheckpoint(Path(directory) / "model.pt", metadata)
            callback.on_epoch_end(0, {"val_loss": 0.2, "model": model, "optimizer": None, "history": {"train": [0.3], "val": [0.2]}})
            bundle = load_orchid_checkpoint(Path(directory) / "model.pt")
            self.assertEqual(bundle["metadata"]["class_labels"], ["Alpha", "Beta"])


if __name__ == "__main__":
    unittest.main()
