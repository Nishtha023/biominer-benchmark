"""
pipeline.py

Orchestrates the full benchmarking run: load data -> run BioMiner on each
paper -> score predictions against ground truth -> collect results.

This module contains no CLI logic (see run_benchmark.py for that) so it can
also be imported and reused directly, e.g. from a notebook.
"""

import os
import time
from typing import List, Optional

import pandas as pd

from .biominer_runner import BioMinerRunner
from .data_loader import PaperGroundTruth, build_paper_dataset
from .metrics import evaluate_paper


def run_pipeline(
    pdf_dir: str,
    ground_truth_csv: str,
    biominer_repo_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the end-to-end benchmarking pipeline and return a results DataFrame
    (one row per paper), which is also saved to <output_dir>/results.csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset: List[PaperGroundTruth] = build_paper_dataset(pdf_dir, ground_truth_csv)
    if not dataset:
        raise RuntimeError(
            "No papers found with both a ground-truth entry and a matching PDF. "
            "Check --pdf_dir and --ground_truth_csv."
        )

    runner = BioMinerRunner(biominer_repo_path, config_path=config_path)

    rows = []
    for paper in dataset:
        print(f"[pipeline] Running BioMiner on paper {paper.paper_id} "
              f"({len(paper.smiles_list)} ground-truth molecules)...")

        paper_output_dir = os.path.join(output_dir, f"paper_{paper.paper_id}")
        start = time.time()
        predicted_smiles = runner.run_on_pdf(paper.pdf_path, paper_output_dir)
        elapsed = time.time() - start

        scores = evaluate_paper(predicted_smiles, paper.smiles_list)
        scores["paper_id"] = paper.paper_id
        scores["processing_time_sec"] = round(elapsed, 2)
        rows.append(scores)

        print(
            f"  -> precision={scores['precision']:.3f} "
            f"recall={scores['recall']:.3f} f1={scores['f1']:.3f} "
            f"mean_tanimoto={scores['mean_best_tanimoto']:.3f} "
            f"({elapsed:.1f}s)"
        )

    results_df = pd.DataFrame(rows)
    # Put paper_id first for readability
    cols = ["paper_id"] + [c for c in results_df.columns if c != "paper_id"]
    results_df = results_df[cols]

    results_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"[pipeline] Saved per-paper results to {results_path}")

    summary_path = os.path.join(output_dir, "summary.csv")
    summary = results_df[["precision", "recall", "f1", "mean_best_tanimoto"]].mean().to_frame().T
    summary.to_csv(summary_path, index=False)
    print(f"[pipeline] Saved aggregate summary to {summary_path}")

    return results_df
