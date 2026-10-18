"""Command line interface: ``pocketnet predict protein.pdb -o results/``."""
import argparse, csv, json, os, sys
from . import __version__
from .api import PocketPredictor


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pocketnet", description="Predict ligand-binding pockets with PocketNet.")
    ap.add_argument("--version", action="version", version=f"pocketnet {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="rank pockets of one or more PDB files")
    p.add_argument("pdb", nargs="+", help="input PDB file(s) (ATOM records; mmCIF is not supported)")
    p.add_argument("-o", "--out", default="pocketnet_out", help="output directory (default: pocketnet_out)")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None, help="default: cuda if available, else cpu")
    p.add_argument("--max-pockets", type=int, default=10)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pred = PocketPredictor(a.device)
    for path in a.pdb:
        res = pred.predict(path, max_pockets=a.max_pockets); stem = os.path.basename(path)
        with open(os.path.join(a.out, stem + "_pockets.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["rank", "score", "probability", "center_x", "center_y", "center_z", "n_points", "residues"])
            for r in res: w.writerow([r["rank"], f"{r['score']:.3f}", f"{r['probability']:.3f}", *r["center"], r["n_points"], " ".join(r["residues"])])
        with open(os.path.join(a.out, stem + "_pockets.json"), "w") as f: json.dump(res, f, indent=1)
        print(f"{stem}: {len(res)} pockets")
        for r in res[:5]: print(f"  {r['rank']:>2}  score {r['score']:6.2f}  p~{r['probability']:.2f}  centre {r['center']}  ({len(r['residues'])} residues)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
