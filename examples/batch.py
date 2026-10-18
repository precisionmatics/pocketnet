"""Predict a folder of structures and collect the first pocket of each in a table."""
import sys, csv
from pathlib import Path
from pocketnet import PocketPredictor

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "examples")
pred = PocketPredictor()
with open("first_pockets.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["structure", "score", "probability", "x", "y", "z"])
    for name, res, err in pred.predict_many(sorted(folder.glob("*.pdb"))):
        if err: print("skipped", name, err); continue
        b = res.best; w.writerow([Path(name).name, f"{b.score:.2f}", f"{b.probability:.2f}", *b.center])
