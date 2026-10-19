# PocketNet

[![PyPI](https://img.shields.io/pypi/v/pocketnet)](https://pypi.org/project/pocketnet/) [![tests](https://github.com/precisionmatics/pocketnet/actions/workflows/ci.yml/badge.svg)](https://github.com/precisionmatics/pocketnet/actions) [![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Ligand-binding pocket prediction from a protein structure. A gated message-passing network over the atom graph feeds an attention-pooling
read-out over the 32 nearest atoms of every solvent-accessible surface point; the scored points are clustered into ranked pockets.
The package bundles a five-model ensemble (about 20 MB of weights): no download is needed at run time, and it runs on CPU or GPU.

```bash
pip install pocketnet
pocketnet predict 3POZ -o results/          # PDB identifier, file, folder or UniProt accession
```

```python
from pocketnet import PocketPredictor

pred = PocketPredictor()                     # CUDA if available, otherwise CPU
result = pred.predict("protein.pdb")         # also mmCIF, "3POZ", "P00533" (AlphaFold DB) or a Structure
print(result)                                # ranked pockets with scores and calibrated probabilities

best = result.best
best.center, best.residues[:5], best.probability
result.residue_scores                        # score of every residue
result.save("out/")                          # CSV, JSON, B-factor PDB, PyMOL (.pml), ChimeraX (.cxc) and SDF
```

## Features
* PDB and mmCIF input (gzip supported), chain selection, PDB identifiers and AlphaFold models fetched on demand
* Ranked pockets with centre, score, calibrated probability, druggability flag, convex-hull volume, lining residues and member surface points
* Per-residue scores and a PDB file whose B-factor column holds them, ready to colour in any viewer
* Exporters for CSV, JSON, pandas DataFrame, SDF (RDKit/OpenBabel-compatible), PyMOL script and ChimeraX command file
* `filter()` for slicing by probability, score or rank; `_repr_html_()` for Jupyter display with probability bars
* Context-manager interface, `residue_scores_df()` for pandas DataFrame of per-residue scores
* Batch prediction that survives bad inputs; typed API (`py.typed`), command line, 20 tests, continuous integration

## Quick examples

### Filtering and volume
```python
result = PocketPredictor().predict("3POZ")

# keep only druggable pockets (probability ≥ 0.5)
druggable = result.filter(druggable_only=True)
for p in druggable:
    print(f"rank {p.rank}  p={p.probability:.2f}  vol={p.volume(result.points):.0f} Å³")

# all volumes in one call
print(result.volumes)
```

### SDF export (RDKit-compatible)
```python
result.write_sdf("pockets.sdf")

# read back with RDKit
from rdkit.Chem import SDMolSupplier
for mol in SDMolSupplier("pockets.sdf", removeHs=False):
    print(mol.GetProp("RANK"), mol.GetProp("PROBABILITY"), mol.GetProp("VOLUME_A3"))
```

### Context manager and residue DataFrame
```python
with PocketPredictor() as pred:
    result = pred.predict("2HHB")
    df = pred.residue_scores_df(result)   # pandas DataFrame, sorted by score
    print(df.head(10))
```

### Jupyter display
```python
result   # calls _repr_html_() automatically — probability bars, druggability ticks, volumes
```

### Batch prediction
```python
import pocketnet
pred = PocketPredictor()
for name, result, err in pred.predict_many(["3POZ", "2HHB", "bad_id"]):
    if err: print(f"{name}: FAILED ({err})")
    else:   print(f"{name}: {len(result)} pockets, best p={result.best.probability:.2f}")
```

## Performance

Evaluated using a 4 Å distance-to-any-ligand-atom criterion, at most 10 pockets per method,
on sequence-identity-filtered subsets of HOLO4K and COACH420 (primary unseen set: < 30 % identity,
≥ 40 % coverage to every training chain).  Numbers are top-*n* / top-*n*+2 / top-10 success rates
(% of ligands found); bootstrap 95 % CIs and McNemar tests are in the manuscript.

**HOLO4K — 415 proteins, 931 ligands (primary unseen)**

| Method | top-*n* | top-*n*+2 | top-10 |
|---|---|---|---|
| **PocketNet (v0.3, rot5 ensemble)** | **71.1** | **82.9** | **91.0** |
| P2Rank | 67.5 | 73.0 | 75.6 |
| DeepPocket | 56.6 | 64.6 | 71.5 |
| PUResNet | 39.0 | 39.2 | 39.2 |
| fpocket | 33.3 | 42.0 | 49.2 |
| Kalasanty | 33.3 | 34.0 | 34.0 |

**COACH420 — 49 proteins, 60 ligands (primary unseen)**

| Method | top-*n* | top-*n*+2 | top-10 |
|---|---|---|---|
| **PocketNet (v0.3, rot5 ensemble)** | **71.7** | **80.0** | **85.0** |
| P2Rank | 70.0 | 71.7 | 73.3 |
| DeepPocket | 60.0 | 73.3 | 80.0 |
| PUResNet | 53.3 | 53.3 | 53.3 |
| Kalasanty | 60.0 | 60.0 | 60.0 |
| fpocket | 35.0 | 48.3 | 70.0 |

v0.3 bundles the **rot5 ensemble**: rotation-augmented models for folds 0–1, standard v3 models for folds 2–4.
Seed variance across three training seeds: ±1–2 pp on HOLO4K (top-n), confirming stability.

Limitations: VN-EGNN, EquiPocket, DeepSurf not benchmarked (no released weights).
Trained on PLAS-20k (simulation-derived complexes); performance on membrane proteins,
large assemblies and cryptic sites is not yet characterised.
Ligands, ions and nucleic acids in the input are ignored.

## Repository layout
`pocketnet/` package and weights, `tests/`, `examples/`, `docs/`, `training/` (data preparation and training), `benchmarks/` (scoring on the P2Rank datasets).

## Citation
Arulsamy S, Natarajan K. PocketNet: message passing and attention pooling on the protein surface for ligand-binding pocket prediction (manuscript in preparation). See `CITATION.cff`.

## License
MIT. See `LICENSE`.
