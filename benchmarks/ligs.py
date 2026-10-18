"""Ground-truth ligands of a P2Rank benchmark dataset, as P2Rank defines them (eval-predict cases/ligands.csv).
Set P2RANK_DATASETS to a clone of https://github.com/rdk/p2rank-datasets."""
import csv, os, numpy as np

ROOT = os.environ.get("P2RANK_DATASETS", "p2rank-datasets")


def load(evaldir):
    rows = list(csv.reader(open(f"{evaldir}/cases/ligands.csv"))); h = [x.strip() for x in rows[0]]
    return rows, h


def coords(path, ids):
    want = set(ids); d = {}
    for ln in open(path):
        if ln[:6] in ("ATOM  ", "HETATM") and int(ln[6:11]) in want:
            d[int(ln[6:11])] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    return np.array([d[i] for i in ids if i in d], np.float32)
