#!/usr/bin/env python3
"""
run_benchmark.py

test 2

Main entry point for benchmarking BioMiner on the XN project's molecular
structure extraction dataset.

Usage:
    python run_benchmark.py \
        --pdf_dir data/pdfs \
        --ground_truth_csv data/ground_truth_SMILES.csv \
        --biominer_repo_path /path/to/BioMiner \
        --output_dir results

No paths are hardcoded -- every input location is passed in via command-line
arguments, so this script runs on any machine as long as the arguments point
to valid locations.

Prerequisites (see README.md for full setup):
    1. BioMiner cloned and installed per its own README
       (https://github.com/jiaxianyan/BioMiner), including downloading its
       model weights (MolDetV2, MolGlyph, BioMiner-Instruct or an API key
       for a closed-source MLLM).
    2. BioMiner's background servers already running (mineru_server.py,
       moldet_server.py, ocsr_server.py, and the vLLM server if using
       BioMiner-Instruct).
    3. This repo's own dependencies installed: pip install -r requirements.txt
"""

import argparse
import sys

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BioMiner on a set of papers with known ground-truth SMILES."
    )
    parser.add_argument(
        "--pdf_dir",
        required=True,
        help="Directory containing paper PDFs, named <paperID>.pdf (e.g. 19.pdf)",
    )
    parser.add_argument(
        "--ground_truth_csv",
        required=True,
        help="CSV with columns: paperID, name of molecule, canonical_SMILES",
    )
    parser.add_argument(
        "--biominer_repo_path",
        required=True,
        help="Path to a local clone of https://github.com/jiaxianyan/BioMiner",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory to write per-paper outputs and the aggregated results CSV",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Optional override for BioMiner's config YAML "
             "(defaults to BioMiner/config/default_open_source.yaml inside the repo)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(
            pdf_dir=args.pdf_dir,
            ground_truth_csv=args.ground_truth_csv,
            biominer_repo_path=args.biominer_repo_path,
            output_dir=args.output_dir,
            config_path=args.config_path,
        )
    except Exception as exc:
        print(f"[run_benchmark] Pipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
