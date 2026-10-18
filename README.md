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
result.save("out/")                          # CSV, JSON, B-factor PDB, PyMOL (.pml) and ChimeraX (.cxc) scripts
```

## Features
* PDB and mmCIF input (gzip supported), chain selection, PDB identifiers and AlphaFold models fetched on demand
* Ranked pockets with centre, score, calibrated probability, lining residues and member surface points
* Per-residue scores and a PDB file whose B-factor column holds them, ready to colour in any viewer
* Exporters for CSV, JSON, pandas, PyMOL and ChimeraX; batch prediction that survives bad inputs
* Typed API (`py.typed`), command line, 20 tests, continuous integration

See the [user guide](docs/usage.md), the [changelog](CHANGELOG.md) and `examples/`.

## Performance (summary)
Evaluated with P2Rank's ligand definitions and a 4 A distance-to-any-ligand-atom criterion, all methods capped at 10 pockets, on HOLO4K and COACH420
after sequence-identity filtering against the training set. PocketNet is on par with P2Rank for the first-ranked pocket, ahead of P2Rank and DeepPocket at larger pocket
budgets, and behind GrASP at top-n under GrASP's own protocol. Tables, confidence intervals, McNemar tests and caveats are in the manuscript.

Limitations: PUResNet was run through our own wrapper; VN-EGNN, EquiPocket, DeepSurf and Kalasanty were not benchmarked; the model was trained on PLAS-20k complexes
(simulation-derived protein-ligand structures), so performance on very different targets (membrane proteins, large assemblies, cryptic sites) is untested.
Ligands, ions and nucleic acids in the input are ignored.

## Repository layout
`pocketnet/` package and weights, `tests/`, `examples/`, `docs/`, `training/` (data preparation and training), `benchmarks/` (scoring on the P2Rank datasets).

## Citation
Arulsamy S, Natarajan K. PocketNet: message passing and attention pooling on the protein surface for ligand-binding pocket prediction (manuscript in preparation). See `CITATION.cff`.

## License
MIT. See `LICENSE`.
