"""Rank the pockets of a structure and write every output format."""
from pocketnet import PocketPredictor

pred = PocketPredictor()                       # CUDA if available, otherwise CPU
result = pred.predict("examples/3POZ.pdb")     # a path, mmCIF, PDB identifier ("3POZ") or UniProt accession ("P00533")
print(result)

best = result.best
print("best pocket centre:", best.center, "probability ~", round(best.probability, 2))
print("lining residues:", best.residues[:8], "...")
top_residues = sorted(result.residue_scores.items(), key=lambda kv: -kv[1])[:5]
print("highest-scoring residues:", top_residues)

result.save("quickstart_out")                  # CSV, JSON, B-factor PDB, PyMOL (.pml) and ChimeraX (.cxc) scripts
