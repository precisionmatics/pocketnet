"""Command line interface: ``pocketnet predict protein.pdb -o results/``."""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from . import __version__

FORMATS = ("csv", "json", "pdb", "pml", "cxc", "sdf")


def _expand(items):
    out = []
    for it in items:
        p = Path(it).expanduser()
        if p.is_dir(): out += sorted(q for q in p.iterdir() if q.suffix.lower() in (".pdb", ".ent", ".cif", ".mmcif", ".gz"))
        else: out.append(it)
    return out


def _predict(a) -> int:
    from .api import PocketPredictor
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(message)s")
    fmts = [f.strip() for f in a.formats.split(",") if f.strip()]; bad = [f for f in fmts if f not in FORMATS]
    if bad: print(f"unknown format(s): {bad}; choose from {FORMATS}", file=sys.stderr); return 2
    items = _expand(a.input)
    if not items: print("no input structures found", file=sys.stderr); return 2
    pred = PocketPredictor(a.device, thr_q=a.top_fraction_q) if a.top_fraction_q else PocketPredictor(a.device); failed = 0
    chains = [c for c in a.chains.split(",") if c] if a.chains else None
    for name, res, err in pred.predict_many(items, chains=chains, max_pockets=a.max_pockets):
        if err is not None: failed += 1; print(f"{name}: FAILED ({err})", file=sys.stderr); continue
        stem = Path(str(name)).name.split(".")[0] if Path(str(name)).exists() else str(name).upper()
        res.save(a.out, stem, fmts)
        if not a.quiet: print(res); print(f"  -> {a.out}/{stem}_*  ({', '.join(fmts)})")
    return 1 if failed and failed == len(items) else 0


def _fetch(a) -> int:
    from .structure import fetch_alphafold, fetch_pdb, looks_like_pdb_id
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for i in a.ids:
        p = fetch_pdb(i, a.out, a.format) if looks_like_pdb_id(i) else fetch_alphafold(i, a.out); print(p)
    return 0


def _view(a) -> int:
    from .api import PocketPredictor
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pred = PocketPredictor(a.device)
    chains = [c for c in a.chains.split(",") if c] if a.chains else None
    res = pred.predict(a.input, chains=chains, max_pockets=a.max_pockets)
    if not a.quiet: print(res)
    html_path = res.view(open_browser=not a.no_browser)
    print(f"  -> viewer: {html_path}")
    return 0


def _info(a) -> int:
    import torch
    from .api import WEIGHTS
    print(f"pocketnet {__version__}"); print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    print(f"{len(WEIGHTS)} bundled models: {', '.join(w.name for w in WEIGHTS)}"); return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pocketnet", description="Predict ligand-binding pockets with PocketNet.")
    ap.add_argument("--version", action="version", version=f"pocketnet {__version__}"); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="rank the pockets of structures (files, folders, PDB identifiers or UniProt accessions)")
    p.add_argument("input", nargs="+", help="PDB/mmCIF file(s), folder(s), PDB identifier(s) such as 3POZ, or UniProt accession(s) for AlphaFold models")
    p.add_argument("-o", "--out", default="pocketnet_out", help="output directory (default: pocketnet_out)")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None, help="default: cuda if available, else cpu")
    p.add_argument("--chains", default="", help="comma-separated chain identifiers to keep, e.g. A,B")
    p.add_argument("--max-pockets", type=int, default=10); p.add_argument("--formats", default=",".join(FORMATS), help=f"comma-separated subset of {', '.join(FORMATS)}")
    p.add_argument("--top-fraction-q", type=float, default=None, help="advanced: quantile threshold for surface points (default 0.97)"); p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(fn=_predict)
    v = sub.add_parser("view", help="predict pockets and open an interactive browser viewer")
    v.add_argument("input", help="PDB/mmCIF file, PDB identifier (e.g. 3POZ) or UniProt accession")
    v.add_argument("--device", choices=["cpu", "cuda"], default=None)
    v.add_argument("--chains", default="", help="comma-separated chain IDs to keep")
    v.add_argument("--max-pockets", type=int, default=10)
    v.add_argument("--no-browser", action="store_true", help="write HTML but do not open browser")
    v.add_argument("-q", "--quiet", action="store_true")
    v.set_defaults(fn=_view)
    f = sub.add_parser("fetch", help="download structures from the PDB or AlphaFold DB"); f.add_argument("ids", nargs="+"); f.add_argument("-o", "--out", default="."); f.add_argument("--format", choices=["pdb", "cif"], default="pdb"); f.set_defaults(fn=_fetch)
    i = sub.add_parser("info", help="show version, device and bundled models"); i.set_defaults(fn=_info)
    a = ap.parse_args(argv); return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
