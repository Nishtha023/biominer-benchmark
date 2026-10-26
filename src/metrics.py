"""
metrics.py

Evaluation metrics for comparing BioMiner's predicted SMILES against the
course's confirmed ground-truth SMILES, at the level of one paper's full
molecule set.

Two metric families are implemented, per the assignment's requirements:

1. Exact match (precision / recall / F1), using RDKit canonicalization so
   that two chemically identical SMILES written differently still count as
   a match (e.g. "CCO" vs "OCC").
2. Tanimoto similarity, using RDKit Morgan fingerprints, to give partial
   credit for structures that are close-but-not-identical -- useful for
   seeing whether a tool is "nearly right" rather than only pass/fail.
"""

from typing import Dict, List, Optional

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs

# Silence RDKit's verbose parsing warnings; we handle invalid SMILES ourselves.
RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles: str) -> Optional[str]:
    """Return the RDKit canonical form of a SMILES string, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def exact_match_metrics(predicted: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """
    Compute set-based precision / recall / F1 for one paper, comparing the
    canonicalized predicted SMILES against the canonicalized ground-truth
    SMILES.

    This mirrors the evaluation approach used in the course's reference
    paper (Gulluoglu et al., 2026): a true positive is a ground-truth
    molecule whose canonical SMILES appears somewhere in the tool's
    predictions; everything predicted beyond that is a false positive;
    everything in ground truth but not predicted is a false negative.
    """
    pred_canonical = {c for c in (canonicalize(s) for s in predicted) if c is not None}
    gt_canonical = {c for c in (canonicalize(s) for s in ground_truth) if c is not None}

    true_positives = pred_canonical & gt_canonical
    false_positives = pred_canonical - gt_canonical
    false_negatives = gt_canonical - pred_canonical

    precision = (
        len(true_positives) / len(pred_canonical) if pred_canonical else 0.0
    )
    recall = len(true_positives) / len(gt_canonical) if gt_canonical else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "num_predicted_valid": len(pred_canonical),
        "num_predicted_total": len(predicted),
        "num_ground_truth": len(gt_canonical),
    }


def _morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def best_tanimoto_match(predicted: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """
    For each ground-truth molecule, find its best Tanimoto similarity against
    any predicted SMILES, then average across the paper. This gives a
    "partial credit" signal for near-miss structures (e.g. one wrong
    substituent) that the strict exact-match metric would score as a
    complete failure.

    Returns:
        dict with "mean_best_tanimoto" (0.0-1.0) and "num_compared"
    """
    pred_fps = [fp for fp in (_morgan_fingerprint(s) for s in predicted) if fp is not None]

    if not pred_fps:
        return {"mean_best_tanimoto": 0.0, "num_compared": len(ground_truth)}

    best_scores = []
    for gt_smiles in ground_truth:
        gt_fp = _morgan_fingerprint(gt_smiles)
        if gt_fp is None:
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(gt_fp, pred_fps)
        best_scores.append(max(similarities) if similarities else 0.0)

    mean_score = sum(best_scores) / len(best_scores) if best_scores else 0.0
    return {"mean_best_tanimoto": mean_score, "num_compared": len(best_scores)}


def evaluate_paper(predicted: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """Convenience wrapper: run both metric families for one paper and merge results."""
    results = exact_match_metrics(predicted, ground_truth)
    results.update(best_tanimoto_match(predicted, ground_truth))
    return results
