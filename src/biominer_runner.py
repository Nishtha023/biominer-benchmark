"""
biominer_runner.py

Thin wrapper around BioMiner's own command-line inference script
(example_open_source.py), so the rest of our pipeline doesn't need to know
anything about how BioMiner is invoked internally.

BioMiner is a separate project (https://github.com/jiaxianyan/BioMiner) that
must be installed and configured independently -- see README.md for setup.
We do not vendor or modify BioMiner's own code here; we only call it as a
subprocess and parse its output, which is how the BioMiner README itself
describes running inference:

    python3 example_open_source.py \
        --config_path=BioMiner/config/default_open_source.yaml \
        --pdf=<path-to-pdf-or-dir> \
        --output_dir=<output-dir>

BioMiner writes its extracted bioactivity triplets (protein, ligand id, type,
value, unit, smiles) to a results file under --output_dir. This wrapper reads
that file back and extracts just the predicted SMILES, since structure
recognition accuracy (not bioactivity-value accuracy) is what our course
benchmark evaluates.
"""

import glob
import os
import subprocess
from typing import List, Optional

import pandas as pd


class BioMinerRunner:
    """Runs BioMiner inference on a single PDF and returns predicted SMILES."""

    def __init__(self, biominer_repo_path: str, config_path: Optional[str] = None):
        """
        Args:
            biominer_repo_path: path to a local clone of the BioMiner repo
                (https://github.com/jiaxianyan/BioMiner). Must contain
                example_open_source.py and have its conda environment(s)
                already set up per BioMiner's own README.
            config_path: path to BioMiner's config YAML. Defaults to
                BioMiner's own default_open_source.yaml inside the repo if
                not provided.
        """
        self.repo_path = biominer_repo_path
        self.script_path = os.path.join(biominer_repo_path, "example_open_source.py")
        self.config_path = config_path or os.path.join(
            biominer_repo_path, "BioMiner", "config", "default_open_source.yaml"
        )

        if not os.path.isfile(self.script_path):
            raise FileNotFoundError(
                f"Could not find example_open_source.py under {biominer_repo_path}. "
                "Did you clone the BioMiner repo and pass the correct path?"
            )

    def run_on_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        Run BioMiner on a single PDF and return the list of predicted SMILES.

        BioMiner must already be running as a background server (per its
        README: mineru_server.py, moldet_server.py, ocsr_server.py, and
        optionally the vLLM server for BioMiner-Instruct) before this is
        called -- this wrapper only triggers the client-side inference call.
        """
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python3",
            self.script_path,
            f"--config_path={self.config_path}",
            f"--pdf={pdf_path}",
            f"--output_dir={output_dir}",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = self.repo_path

        result = subprocess.run(
            cmd, cwd=self.repo_path, env=env, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[biominer_runner] BioMiner failed on {pdf_path}:\n{result.stderr}")
            return []

        return self._extract_smiles_from_output(output_dir)

    @staticmethod
    def _extract_smiles_from_output(output_dir: str) -> List[str]:
        """
        Parse BioMiner's result file(s) in output_dir and pull out the
        'smiles' column. BioMiner's own README shows its output as a table
        with columns: protein, ligand, type, value, unit, smiles -- saved
        as a CSV/TSV under the output directory.
        """
        candidate_files = glob.glob(os.path.join(output_dir, "**", "*.csv"), recursive=True)
        candidate_files += glob.glob(os.path.join(output_dir, "**", "*.tsv"), recursive=True)

        all_smiles: List[str] = []
        for file_path in candidate_files:
            try:
                sep = "\t" if file_path.endswith(".tsv") else ","
                df = pd.read_csv(file_path, sep=sep)
            except Exception as exc:  # pragma: no cover - defensive parsing
                print(f"[biominer_runner] Could not parse {file_path}: {exc}")
                continue

            if "smiles" in df.columns:
                all_smiles.extend(df["smiles"].dropna().astype(str).tolist())

        # De-duplicate while preserving order
        seen = set()
        deduped = []
        for s in all_smiles:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped
