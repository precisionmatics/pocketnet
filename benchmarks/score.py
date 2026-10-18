"""Ligand-level DCA success (P2Rank convention): a protein with n ligands is allowed n (top-n), n+2 or 10 pockets;
a ligand counts as found if one allowed pocket centre lies within 4 A of any of its atoms. All methods are capped at 10."""
import numpy as np


def score_protein(centers, ligs, thr=4.0, cap=10):
    n = len(ligs); out = {}
    for k, name in ((0, "top_n"), (2, "top_n+2"), (999, "top_10")):
        c = centers[:min(n + k, cap)] if len(centers) else np.zeros((0, 3))
        out[name] = [float(len(c) > 0 and (np.linalg.norm(c[:, None, :] - l[None], axis=-1).min() <= thr)) for l in ligs]
    return out


def aggregate(rows):
    return {k: float(np.mean([v for r in rows for v in r[k]])) for k in ("top_n", "top_n+2", "top_10")} | {
        "ligands": sum(len(r["top_n"]) for r in rows), "proteins": len(rows)}
