# PocketNet user guide

## Inputs
`PocketPredictor.predict` and `pocketnet predict` accept

| Input | Example |
|---|---|
| PDB file | `protein.pdb`, `protein.ent`, `protein.pdb.gz` |
| mmCIF file | `protein.cif`, `protein.cif.gz` (author chain and residue numbering) |
| PDB-format text | the contents of a PDB file as a string |
| PDB identifier | `3POZ` (downloaded from RCSB, cached in `~/.cache/pocketnet`) |
| UniProt accession | `P00533` (AlphaFold DB model, cached) |
| `Structure` object | `pocketnet.read_structure(...)` |

Only `ATOM` records of the first model are used (plus selenomethionine); hydrogens, water, ligands, ions and alternate locations other than A are ignored.
For AlphaFold models the B-factor column holds pLDDT, available as `result.structure.bfactor`.

## Python
```python
from pocketnet import PocketPredictor
pred = PocketPredictor(device="cpu")            # or "cuda"; extraction settings can be overridden: PocketPredictor(link=8.0)
res = pred.predict("3POZ", chains=["A"], max_pockets=5)

res.best.center                                 # [x, y, z]
res.best.residues                               # ['A:718', 'A:726', ...]
res.residue_scores                              # {'A:790': 0.97, ...} highest nearby surface-point score per residue
res.points, res.point_scores                    # the solvent-accessible surface points and their scores
res.to_dataframe()                              # pandas (optional dependency)
res.save("out/", formats=("csv", "json", "pdb", "pml", "cxc"))

for name, r, err in pred.predict_many(["a.pdb", "b.pdb"]):   # keeps going after a failure
    ...
```

## Command line
```bash
pocketnet predict protein.pdb -o results/
pocketnet predict structures/ --chains A,B --max-pockets 5 --formats csv,json
pocketnet predict 3POZ P00533 -o results/        # fetch from the PDB and AlphaFold DB
pocketnet fetch 3POZ --format cif -o data/
pocketnet info
```
Outputs per structure: `*_pockets.csv`, `*_pockets.json`, `*_pocketnet.pdb` (occupancy = rank of the best pocket containing the residue, B-factor = 100 x residue score,
pocket centres as `PKT` pseudo-atoms), `*_pockets.pml` (PyMOL) and `*_pockets.cxc` (ChimeraX).

## What the numbers mean
* `score`: sum of the 24 highest surface-point scores of the pocket; use it to order pockets.
* `probability`: isotonic calibration of `score` on held-out proteins of one fold. Treat it as a guide to ordering, not as an exact probability.
* A pocket is counted as correct in the paper when its centre lies within 4 A of a ligand atom.

## Limits
Structures above about 14,000 heavy atoms were not seen in training. Nucleic acids, ligands and ions in the input are ignored. PocketNet predicts *where* a ligand could bind, not which
ligand binds. Benchmarks and caveats are in the manuscript.
