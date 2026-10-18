# Changelog

## 0.2.0
* New `PredictionResult` and `Pocket` objects (dictionary-style access from 0.1 still works) with per-residue scores and the surface points.
* Exporters: CSV, JSON, B-factor PDB, PyMOL and ChimeraX scripts, pandas DataFrame.
* mmCIF and gzip input, chain selection, insertion codes, PDB identifiers (RCSB) and UniProt accessions (AlphaFold DB) as input.
* `predict_many`, configurable extraction settings, logging, type information (`py.typed`).
* Command line: folders as input, `--chains`, `--formats`, `fetch` and `info` commands.
* Tests (20), continuous integration, user guide.

## 0.1.0
* First release: five-model ensemble with bundled weights, `PocketPredictor`, `pocketnet predict`.
