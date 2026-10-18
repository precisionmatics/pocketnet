# Training and data preparation

These scripts are provided for transparency and reproduction; they are not needed to use the package.

1. `extract.py`: streams the PLAS-20K tarballs (`PLAS20K_DIR`) into per-complex `.npz` files (`RAW_DIR`, default `data/raw`).
2. `build_cache.py K0 K1`: featurises the complexes on GPU into `data/cache` (run K1 workers with K0 = 0..K1-1).
3. `train.py --meta META.pkl --fold F --name NAME`: trains one fold model. `META.pkl` is a pickle with `ids` (list of complex ids)
   and `folds[i]["role"]` (a train/val/test label per id). The split used in the paper clusters proteins with MMseqs2 (30 % and 80 % identity)
   and groups ligands by RDKit scaffold; the split file itself is not bundled.

The released weights are the five fold models (`v3`), trained with the defaults in `train.py` plus `--full`.
