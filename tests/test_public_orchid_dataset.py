import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from model_compression.src.orchid.public_dataset import (
    select_from_aws_metadata,
    validate_manifests,
    write_manifests,
)


class PublicOrchidDatasetTests(unittest.TestCase):
    def write_gzip_csv(self, path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
        with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def test_selects_one_licensed_photo_per_observation_and_writes_four_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata"
            metadata.mkdir()
            taxa = [
                {"taxon_id": "1", "name": "Orchidaceae", "rank": "family", "ancestry": "48460"},
                {"taxon_id": "2", "name": "Cattleya trianae", "rank": "species", "ancestry": "48460/1"},
                {"taxon_id": "3", "name": "Dendrobium nobile", "rank": "species", "ancestry": "48460/1"},
            ]
            observations = []
            photos = []
            for species_taxon, prefix in (("2", "cat"), ("3", "den")):
                for number in range(4):
                    observation = f"{prefix}-obs-{number}"
                    observations.append({"observation_uuid": observation, "taxon_id": species_taxon, "quality_grade": "research", "observer_id": f"user-{number}"})
                    photos.append({"observation_uuid": observation, "photo_id": f"{prefix}-{number}-a", "extension": "jpg", "license": "cc-by", "observer_id": f"user-{number}", "position": "0"})
                    photos.append({"observation_uuid": observation, "photo_id": f"{prefix}-{number}-b", "extension": "jpg", "license": "cc-by-nc", "observer_id": f"user-{number}", "position": "1"})
            self.write_gzip_csv(metadata / "taxa.csv.gz", list(taxa[0]), taxa)
            self.write_gzip_csv(metadata / "observations.csv.gz", list(observations[0]), observations)
            self.write_gzip_csv(metadata / "photos.csv.gz", list(photos[0]), photos)
            config = {
                "dataset_id": "test",
                "output_root": str(root / "dataset"),
                "source": {"provider": "inaturalist-licensed-observation-images", "family_name": "Orchidaceae", "quality_grade": "research", "accepted_photo_licenses": ["cc-by"]},
                "selection": {"target_images": 8, "target_species": 2, "minimum_images_per_species": 4, "maximum_images_per_species": 4, "maximum_images_per_observer_per_species": 1, "image_size": "medium", "deterministic_selection_seed": 7},
                "splits": {"train": 0.5, "validation": 0.25, "calibration": 0.125, "test": 0.125, "seed": 17},
                "outputs": {"manifest_directory": "manifests", "observations": "observations.csv", "images": "images.csv", "split": "split.csv", "taxonomy": "taxonomy.json", "licenses": "licenses.csv", "checksums": "checksums.csv", "rejected_records": "rejected_records.csv", "dataset_card": "dataset_card.json"},
            }
            records, summary = select_from_aws_metadata(metadata, config)
            self.assertEqual(8, len(records))
            self.assertEqual({"cc-by"}, {record.photo_license for record in records})
            self.assertEqual(8, len({record.observation_id for record in records}))
            manifest = write_manifests(config["output_root"], records, config, summary)
            self.assertEqual({"images": 8, "species": 2, "genera": 2}, validate_manifests(manifest, config))
            card = json.loads((manifest / "dataset_card.json").read_text(encoding="utf-8"))
            self.assertEqual("manifest_only", card["download_status"])


if __name__ == "__main__":
    unittest.main()
