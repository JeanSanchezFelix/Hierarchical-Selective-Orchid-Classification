"""Coverage-gated phylogenetic distance metrics for orchid predictions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


PEREZ_ESCOBAR_2024 = {
    "citation": "Pérez-Escobar et al. (2024), The origin and speciation of orchids, New Phytologist 242:700–716.",
    "article_doi": "10.1111/nph.19580",
    "data_doi": "10.6084/m9.figshare.22245940.v1",
    "tree_archive": "10.PPtrees.AAR.rar",
    "tree_archive_md5": "64b3dc228074297e0b516d7ce6a3d229",
    "license": "CC BY 4.0",
}


@dataclass
class _Node:
    name: str | None = None
    length: float = 0.0
    children: list["_Node"] | None = None
    parent: "_Node | None" = None


def parse_newick(text: str) -> _Node:
    """Parse a branch-length Newick tree without an additional bioinformatics dependency."""
    tokens = text.strip().rstrip(";")
    position = 0

    def parse_label() -> tuple[str | None, float]:
        nonlocal position
        start = position
        while position < len(tokens) and tokens[position] not in ",():":
            position += 1
        name = tokens[start:position].strip() or None
        length = 0.0
        if position < len(tokens) and tokens[position] == ":":
            position += 1
            length_start = position
            while position < len(tokens) and tokens[position] not in ",()":
                position += 1
            length = float(tokens[length_start:position])
        return name, length

    def parse_subtree() -> _Node:
        nonlocal position
        if position < len(tokens) and tokens[position] == "(":
            position += 1
            children = []
            while True:
                child = parse_subtree()
                children.append(child)
                if position >= len(tokens) or tokens[position] == ")":
                    break
                if tokens[position] != ",":
                    raise ValueError("Malformed Newick tree.")
                position += 1
            position += 1
            name, length = parse_label()
            node = _Node(name=name, length=length, children=children)
            for child in children:
                child.parent = node
            return node
        name, length = parse_label()
        if name is None:
            raise ValueError("Newick leaf has no name.")
        return _Node(name=name, length=length, children=[])

    root = parse_subtree()
    if position != len(tokens):
        raise ValueError("Unexpected content after Newick tree.")
    return root


def _leaves(root: _Node) -> dict[str, _Node]:
    result = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(node.children)
        elif node.name:
            if node.name in result:
                raise ValueError(f"Duplicate tree tip '{node.name}'.")
            result[node.name] = node
    return result


def _path_to_root(node: _Node) -> dict[int, tuple[_Node, float]]:
    distance = 0.0
    path = {}
    current = node
    while current is not None:
        path[id(current)] = (current, distance)
        distance += current.length
        current = current.parent
    return path


def patristic_distance(left: _Node, right: _Node) -> float:
    left_path = _path_to_root(left)
    distance = 0.0
    current = right
    while current is not None:
        if id(current) in left_path:
            return distance + left_path[id(current)][1]
        distance += current.length
        current = current.parent
    raise RuntimeError("Tree nodes have no common ancestor.")


@dataclass(frozen=True)
class PhylogenyCoverage:
    requested_species: int
    mapped_species: int
    unmapped_species_ids: tuple[str, ...]

    @property
    def ratio(self) -> float:
        return self.mapped_species / self.requested_species if self.requested_species else 0.0


def read_species_mapping(path: str | Path) -> dict[str, str]:
    """Read reviewed mappings with `mapping_status=matched`; never guess synonyms."""
    mappings: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("mapping_status", "").strip().lower() != "matched":
                continue
            species_id = row.get("species_id", "").strip()
            source_tip = row.get("source_tip", "").strip()
            if not species_id or not source_tip:
                raise ValueError("Matched phylogeny rows require species_id and source_tip.")
            if species_id in mappings:
                raise ValueError(f"Duplicate mapping for {species_id}.")
            mappings[species_id] = source_tip
    return mappings


def build_distance_matrix(
    newick_path: str | Path,
    species_ids: Iterable[str],
    mapping_path: str | Path,
    minimum_coverage: float = 0.90,
) -> tuple[list[str], np.ndarray, PhylogenyCoverage]:
    """Return normalized distances only when reviewed mapping coverage is sufficient."""
    requested = list(species_ids)
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1].")
    tree_tips = _leaves(parse_newick(Path(newick_path).read_text(encoding="utf-8")))
    mapping = read_species_mapping(mapping_path)
    labels = [species_id for species_id in requested if mapping.get(species_id) in tree_tips]
    coverage = PhylogenyCoverage(len(requested), len(labels), tuple(species_id for species_id in requested if species_id not in labels))
    if coverage.ratio < minimum_coverage:
        raise ValueError(f"Phylogenetic coverage {coverage.ratio:.1%} is below required {minimum_coverage:.1%}; do not report the metric.")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for row, species_id in enumerate(labels):
        for column in range(row + 1, len(labels)):
            distance = patristic_distance(tree_tips[mapping[species_id]], tree_tips[mapping[labels[column]]])
            matrix[row, column] = matrix[column, row] = distance
    maximum = matrix.max()
    if maximum > 0:
        matrix /= maximum
    return labels, matrix, coverage


def build_posterior_distance_summary(
    tree_directory: str | Path,
    species_ids: Iterable[str],
    mapping_path: str | Path,
    minimum_coverage: float = 0.90,
) -> tuple[list[str], np.ndarray, np.ndarray, PhylogenyCoverage]:
    """Summarize distances across every released posterior tree.

    Reporting the posterior mean and standard deviation avoids selecting a single
    convenient topology from the ten published trees.
    """
    trees = sorted(Path(tree_directory).glob("*.tre"))
    if not trees:
        raise FileNotFoundError(f"No .tre files found under {tree_directory}.")
    reference_labels: list[str] | None = None
    matrices = []
    reference_coverage: PhylogenyCoverage | None = None
    for tree in trees:
        labels, matrix, coverage = build_distance_matrix(tree, species_ids, mapping_path, minimum_coverage)
        if reference_labels is None:
            reference_labels, reference_coverage = labels, coverage
        elif labels != reference_labels:
            raise ValueError("Mapped labels differ across posterior trees; inspect source-tip names.")
        matrices.append(matrix)
    stacked = np.stack(matrices)
    return reference_labels or [], stacked.mean(axis=0), stacked.std(axis=0), reference_coverage or PhylogenyCoverage(0, 0, ())


def mean_phylogenetic_error(y_true: Iterable[str], y_pred: Iterable[str], labels: list[str], matrix: np.ndarray) -> float:
    """Mean normalized evolutionary distance between true and predicted labels."""
    index = {label: position for position, label in enumerate(labels)}
    distances = []
    for truth, prediction in zip(y_true, y_pred):
        if truth not in index or prediction not in index:
            raise ValueError("Predictions must be restricted to phylogenetically mapped labels.")
        distances.append(float(matrix[index[truth], index[prediction]]))
    if not distances:
        raise ValueError("At least one prediction is required.")
    return float(np.mean(distances))
