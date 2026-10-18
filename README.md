# PocketNet

Ligand-binding pocket prediction from a protein structure. A gated message-passing network over the atom graph feeds an
attention-pooling read-out over the 32 nearest atoms of each solvent-accessible surface point; surface points are scored,
clustered and ranked into pockets. The package bundles a five-model ensemble (about 20 MB of weights); no download is needed at run time.

```bash
pip install pocketnet
pocketnet predict protein.pdb -o results/
```

```python
from pocketnet import PocketPredictor

pred = PocketPredictor()               # uses the GPU if available, otherwise CPU
for p in pred.predict("protein.pdb"):  # at most 10 pockets, best first
    print(p["rank"], p["score"], p["probability"], p["center"], p["residues"][:5])
```

Each pocket has `rank`, `score` (sum of its 24 highest surface-point scores), `probability` (isotonic calibration of that score on held-out
proteins; indicative only), `center` (x, y, z in angstrom), `n_points` and `residues` (`chain:number` within 8 A of the centre).
The CLI writes `<input>_pockets.csv` and `.json`. A run takes about 1 s per protein on a GPU and a few seconds on a CPU.

## Input
PDB-format files with `ATOM` records (standard residues; hydrogens and alternate locations other than A are ignored, first model only).
mmCIF is not supported. Structures above about 14,000 atoms were not seen in training.

## Performance (summary)
Evaluated with P2Rank's ligand definitions and a 4 A distance-to-any-ligand-atom criterion, all methods capped at 10 pockets, on COACH420, HOLO4K
and Chen11 after sequence- and structure-identity filtering against the training set. PocketNet is on par with P2Rank for the first-ranked pocket,
ahead of P2Rank at larger pocket budgets, and behind GrASP at top-n on some sets. Full tables, confidence intervals, McNemar tests and caveats are in the manuscript.

Limitations: PUResNet was run through our own wrapper; VN-EGNN, EquiPocket, DeepSurf and Kalasanty were not benchmarked; the model was trained on
PLAS-20K complexes (simulation-derived protein-ligand structures), so performance on very different targets (membrane proteins, large assemblies, cryptic sites) is untested.

## Repository layout
`pocketnet/` package and weights, `tests/`, `examples/3POZ.pdb`, `training/` (data preparation and training), `benchmarks/` (scoring on the P2Rank datasets).

## Citation
Arulsamy S, Natarajan K. PocketNet (manuscript in preparation). Please check the repository for an updated citation.

## License
MIT. See `LICENSE`.
