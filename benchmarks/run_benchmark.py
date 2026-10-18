"""Score PocketNet on a P2Rank dataset with P2Rank's ligand definitions.

usage: python run_benchmark.py DATASET_SUBDIR EVALDIR OUT.json [--device cpu|cuda]
  DATASET_SUBDIR  folder inside $P2RANK_DATASETS holding the PDB files (e.g. holo4k)
  EVALDIR         directory of a P2Rank eval-predict run for that dataset (contains cases/ligands.csv)
"""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocketnet import PocketPredictor
from ligs import load, coords, ROOT
from score import score_protein, aggregate

ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("evaldir"); ap.add_argument("out")
ap.add_argument("--device", default=None); a = ap.parse_args()
rows, h = load(a.evaldir); by = {}
for r in rows[1:]: by.setdefault(r[0].strip(), []).append([int(x) for x in r[h.index("atomIds")].split()])
pred = PocketPredictor(a.device); per = {}
for pdb, ids in by.items():
    path = f"{ROOT}/{a.dataset}/{pdb}"
    ligs = [l for l in (coords(path, i) for i in ids) if len(l)]
    if not ligs: continue
    try: pk = pred.predict(path)
    except Exception as e: print("skip", pdb, e); continue
    per[pdb] = score_protein(np.array([p["center"] for p in pk]).reshape(-1, 3), ligs)
out = aggregate(list(per.values())); print(a.dataset, out)
json.dump(dict(agg=out, per=per), open(a.out, "w"))
