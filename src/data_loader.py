"""
data_loader.py

Loads the benchmark dataset used for this project: a set of scientific-paper
PDFs plus a ground-truth CSV mapping each paper to the molecules it contains
and their canonical SMILES strings.

Expected ground-truth CSV format (as provided by the course XN project):
    paperID, name of molecule, canonical_SMILES

Expected PDF directory format:
    <pdf_dir>/<paperID>.pdf   e.g. data/pdfs/19.pdf

No paths are hardcoded here -- callers pass in the directory/file locations
explicitly (see run_benchmark.py for the CLI that wires this together).
"""

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PaperGroundTruth:
    """Ground truth for a single paper: its PDF path and known molecules."""
    paper_id: str
    pdf_path: str
    molecule_names: List[str] = field(default_factory=list)
    smiles_list: List[str] = field(default_factory=list)


def load_ground_truth(csv_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Read the ground-truth CSV and group rows by paperID.

    Returns:
        dict mapping paperID (str) -> list of {"name": ..., "smiles": ...} dicts
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"paperID", "name of molecule", "canonical_SMILES"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Ground truth CSV is missing expected columns. "
                f"Expected at least {required_cols}, got {reader.fieldnames}"
            )
        for row in reader:
            paper_id = row["paperID"].strip()
            grouped[paper_id].append(
                {
                    "name": row["name of molecule"].strip(),
                    "smiles": row["canonical_SMILES"].strip(),
                }
            )

    return dict(grouped)


def build_paper_dataset(pdf_dir: str, ground_truth_csv: str) -> List[PaperGroundTruth]:
    """
    Combine the ground-truth CSV with the PDFs actually present on disk.

    Only papers that have BOTH a ground-truth entry AND a corresponding PDF
    file are included, so the benchmark never silently skips a paper without
    saying why.

    Returns:
        list of PaperGroundTruth objects, one per paper, ready to feed into
        the BioMiner runner.
    """
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    grouped = load_ground_truth(ground_truth_csv)
    dataset: List[PaperGroundTruth] = []
    skipped: List[str] = []

    for paper_id, entries in grouped.items():
        pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
        if not os.path.isfile(pdf_path):
            skipped.append(paper_id)
            continue
        dataset.append(
            PaperGroundTruth(
                paper_id=paper_id,
                pdf_path=pdf_path,
                molecule_names=[e["name"] for e in entries],
                smiles_list=[e["smiles"] for e in entries],
            )
        )

    if skipped:
        print(
            f"[data_loader] WARNING: {len(skipped)} paper(s) have ground truth "
            f"but no matching PDF in {pdf_dir}: {skipped}"
        )

    return dataset
